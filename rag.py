from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any

import chromadb

from RAG.chunker import Chunk, chunk_sections
from RAG.pre_injection.config import load_config
from RAG.pdf_parser import parse_pdf

log = logging.getLogger(__name__)

# Sentinel for "caller didn't override this field; take it from the profile".
# (None can't mean "unset" because None is a valid query_prompt_name.)
_UNSET = object()


@dataclass(frozen=True)
class EmbeddingProfile:
    """A per-platform embedding configuration.

    Bundles everything that has to stay consistent for one embedding model:
    the model, the query prompt it expects, the chunk size that suits it, and
    the embed batch size. Vectors from different models aren't comparable, so
    ingest() and retrieve() on a machine share ONE profile — picked by device.

    Token units differ on purpose:
      * `model_max_tokens` is the model's hard context window, counted in the
        MODEL's own tokens. Input beyond it is silently truncated, so it is the
        binding limitation when choosing `max_chunk_tokens`.
      * `max_chunk_tokens` is the chunker's budget, counted in cl100k tokens
        (see chunker.count_tokens). Dense legal text retokenizes to ~1.3-1.8x
        more model tokens, so keep `max_chunk_tokens` comfortably below
        `model_max_tokens`. ingest() also runs a live truncation check.
    """

    name: str                      # short id used for selection/logging
    platform: str                  # human label for the machine this targets
    model_name: str                # Hugging Face model id
    model_max_tokens: int          # model context window (limitation)
    max_chunk_tokens: int          # chunker budget, in cl100k tokens
    embed_batch_size: int          # chunks per embedding forward pass
    query_prompt_name: str | None  # named query prompt, or None if symmetric
    notes: str = ""


# Apple MacBook — Metal (MPS) or an NVIDIA CUDA GPU. The 0.6B harrier model is
# high quality and fast on GPU/MPS native half precision. batch_size=1 because
# MPS can't fit a batched padded sequence of long chunks (attention is
# O(seq_len^2)); 2048-token chunks keep each section mostly whole.
_GPU_PROFILE = EmbeddingProfile(
    name="gpu",
    platform="Apple MacBook (MPS) / CUDA GPU",
    model_name="microsoft/harrier-oss-v1-0.6b",
    model_max_tokens=8192,
    max_chunk_tokens=2048,
    embed_batch_size=1,
    query_prompt_name="web_search_query",
    notes="0.6B model; high quality, painfully slow on CPU.",
)

# Windows PC / any CPU-only box. ModernBERT is ~4x smaller than harrier, so CPU
# ingestion takes minutes instead of hours, and its 8192-token window embeds
# even long defined-term chunks WHOLE (no truncation, unlike a 512-ctx model).
# max_chunk_tokens is kept small purely for speed — embedding attention is
# O(seq_len^2). Symmetric model -> no query prompt.
_CPU_PROFILE = EmbeddingProfile(
    name="cpu",
    platform="Windows PC / CPU",
    model_name="Alibaba-NLP/gte-modernbert-base",
    model_max_tokens=8192,
    max_chunk_tokens=512,
    embed_batch_size=16,
    query_prompt_name=None,
    notes="149M ModernBERT; minutes on CPU, long window avoids truncation.",
)

EMBEDDING_PROFILES: dict[str, EmbeddingProfile] = {
    p.name: p for p in (_GPU_PROFILE, _CPU_PROFILE)
}

# Which profile to use when embedding_profile="auto", keyed by resolved device.
_DEVICE_DEFAULT_PROFILE = {"cuda": "gpu", "mps": "gpu", "cpu": "cpu"}


class CLOIndentureRAG:
    def __init__(
        self,
        persist_dir: str = "RAG/chroma_db",
        collection_name: str = "clo_indentures",
        reranker_model_name: str = "jinaai/jina-reranker-v3",
        device: str = "auto",
        embedding_profile: str = "auto",
        # The four fields below default to the resolved profile's values; pass
        # one explicitly to override just that field on top of the profile.
        embedding_model_name: Any = _UNSET,
        query_prompt_name: Any = _UNSET,
        max_chunk_tokens: Any = _UNSET,
        embed_batch_size: Any = _UNSET,
        chunk_overlap_tokens: int = 128,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.reranker_model_name = reranker_model_name
        self.device = device
        self.embedding_profile = embedding_profile
        self.chunk_overlap_tokens = chunk_overlap_tokens

        # Per-field overrides applied on top of the resolved profile.
        self._overrides: dict[str, Any] = {}
        for key, value in (
            ("model_name", embedding_model_name),
            ("query_prompt_name", query_prompt_name),
            ("max_chunk_tokens", max_chunk_tokens),
            ("embed_batch_size", embed_batch_size),
        ):
            if value is not _UNSET:
                self._overrides[key] = value

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, embedding_function=None
        )

        self._embedder = None
        self._reranker = None
        # Serializes retrieve() calls so concurrent tool calls don't race
        # on the Jina reranker's HF fast tokenizer (Rust-backed, NOT
        # thread-safe — raises "Already borrowed" on concurrent use) or
        # on lazy model init.
        self._retrieve_lock = threading.Lock()

    @cached_property
    def resolved_device(self) -> str:
        """The device the embedder runs on: 'cuda', 'mps', 'cpu', or whatever
        was passed explicitly. Detected once (importing torch is not free)."""
        return self._resolve_device()

    @cached_property
    def profile(self) -> EmbeddingProfile:
        """Resolve the embedding profile once: pick `embedding_profile` if set,
        else the per-device default, then apply any constructor overrides."""
        if self.embedding_profile != "auto":
            name = self.embedding_profile
        else:
            name = _DEVICE_DEFAULT_PROFILE.get(self.resolved_device)
            if name is None:
                raise ValueError(
                    f"no default profile for device {self.resolved_device!r}; "
                    f"pass embedding_profile= one of {sorted(EMBEDDING_PROFILES)}"
                )
        if name not in EMBEDDING_PROFILES:
            raise ValueError(
                f"unknown embedding_profile {name!r}; "
                f"choose from {sorted(EMBEDDING_PROFILES)}"
            )
        prof = EMBEDDING_PROFILES[name]
        return replace(prof, **self._overrides) if self._overrides else prof

    # Resolved values, read from the profile so external callers and retrieve()
    # see the same model/prompt/chunking the embedder actually uses.
    @property
    def embedding_model_name(self) -> str:
        return self.profile.model_name

    @property
    def query_prompt_name(self) -> str | None:
        return self.profile.query_prompt_name

    @property
    def max_chunk_tokens(self) -> int:
        return self.profile.max_chunk_tokens

    @property
    def embed_batch_size(self) -> int:
        return self.profile.embed_batch_size

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            # On CPU, half-precision (bf16/fp16) matmuls are emulated and run
            # ~20x slower than float32, which uses optimized oneDNN/MKL kernels.
            # GPU/MPS keep "auto" since their native half precision is fast.
            dtype = "auto" if self.resolved_device in ("cuda", "mps") else "float32"
            # trust_remote_code stays on so a custom-arch model can be dropped
            # into a profile later; the built-in profiles load natively.
            self._embedder = SentenceTransformer(
                self.profile.model_name,
                device=self.resolved_device,
                model_kwargs={"dtype": dtype},
                trust_remote_code=True,
            )
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                self.reranker_model_name,
                dtype="auto",
                trust_remote_code=True,
            )
            model.eval()
            self._reranker = model
        return self._reranker

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _warn_if_truncating(self, texts: list[str]) -> None:
        """Guard the 'never truncate raw text' invariant.

        The embedder silently truncates any input past its context window, so
        we tokenize with the model's own tokenizer and warn loudly if a chunk
        would lose its tail. The fix is always a smaller `max_chunk_tokens` or
        a longer-context profile — never silent truncation.
        """
        window = self.embedder.max_seq_length
        tokenizer = self.embedder.tokenizer
        over = [n for n in (len(tokenizer.encode(t)) for t in texts) if n > window]
        if over:
            log.warning(
                "ingest: %d/%d chunks exceed the %d-token model window "
                "(largest=%d) and WILL be truncated — lower max_chunk_tokens "
                "or switch to a longer-context profile",
                len(over), len(texts), window, max(over),
            )
        else:
            log.info(
                "ingest: all %d chunks fit the %d-token window (no truncation)",
                len(texts), window,
            )

    def ingest(
        self,
        pdf_path: str,
        doc_id: str | None = None,
        overwrite: bool = False,
        config_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if doc_id is None:
            doc_id = self._hash_file(pdf_path)

        if config_path is not None:
            config = load_config(config_path)
            log.info("ingest: using config from %s", config_path)
        else:
            try:
                config = load_config(doc_id)
                log.info("ingest: using per-doc config pre_injection/configs/%s.yaml", doc_id)
            except FileNotFoundError:
                config = load_config("_defaults")
                log.warning(
                    "ingest: no per-doc config for %s; using _defaults "
                    "(run `python -m RAG.pre_injection.configure` to generate one)",
                    doc_id,
                )

        log.info(
            "ingest: pdf=%s doc_id=%s overwrite=%s",
            pdf_path, doc_id, overwrite,
        )

        existing = self._collection.get(where={"doc_id": doc_id}, limit=1)
        already_ingested = bool(existing["ids"])

        if already_ingested and not overwrite:
            n_existing = len(
                self._collection.get(where={"doc_id": doc_id})["ids"]
            )
            log.info("ingest: skipped (already %d chunks)", n_existing)
            return {
                "status": "skipped",
                "doc_id": doc_id,
                "n_chunks": n_existing,
            }

        if already_ingested and overwrite:
            log.info("ingest: overwrite=True, deleting existing chunks")
            self._collection.delete(where={"doc_id": doc_id})

        log.info("ingest: parsing PDF...")
        t0 = time.time()
        sections = parse_pdf(pdf_path, config)
        log.info("ingest: parsed %d sections in %.2fs", len(sections), time.time() - t0)

        prof = self.profile
        log.info(
            "ingest: profile=%s (%s) model=%s device=%s max_chunk_tokens=%d batch_size=%d",
            prof.name, prof.platform, prof.model_name, self.resolved_device,
            prof.max_chunk_tokens, prof.embed_batch_size,
        )

        t1 = time.time()
        chunks: list[Chunk] = chunk_sections(
            sections,
            doc_id=doc_id,
            config=config,
            max_tokens=prof.max_chunk_tokens,
            chunk_overlap_tokens=self.chunk_overlap_tokens,
        )
        log.info("ingest: built %d chunks in %.2fs", len(chunks), time.time() - t1)

        if not chunks:
            log.info("ingest: no chunks produced, returning empty result")
            return {
                "status": "empty",
                "doc_id": doc_id,
                "n_chunks": 0,
                "n_sections": len(sections),
            }

        texts = [c["text"] for c in chunks]
        self._warn_if_truncating(texts)
        log.info("ingest: embedding %d chunks...", len(texts))
        t2 = time.time()
        embeddings = self.embedder.encode(
            texts,
            batch_size=prof.embed_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()
        log.info("ingest: embedded in %.2fs", time.time() - t2)

        self._collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[_chroma_safe(c["metadata"]) for c in chunks],
        )
        log.info("ingest: added %d chunks to collection", len(chunks))

        return {
            "status": "ingested",
            "doc_id": doc_id,
            "n_chunks": len(chunks),
            "n_sections": len(sections),
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_rerank: bool = False,
        top_j: int = 5,
        max_definition_chunks: int | None = None,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks for `query`.

        `top_k` is always the embedding-stage count — how many candidates
        the vector search returns. `top_j` only matters when reranking:
        it is the post-rerank final count.

        Modes:
          - `use_rerank=False` (default): return the top_k embedding hits.
            Fast — no cross-encoder cost.
          - `use_rerank=True`: embedding pulls top_k candidates, the cross-
            encoder reranker scores them, and the top_j are returned.
            Recommended call: top_k=12, top_j=5.

        `max_definition_chunks`: caps how many chunk_type="defined_term"
        chunks appear in the final result. Short defined-term chunks tend
        to dominate pure-embedding rankings; capping them frees slots for
        longer body sections that explain how the term is used. Only
        active when use_rerank=False. None disables the cap.

        Each result has a `score` field (rerank score if reranked, cosine
        similarity `1 - embed_distance` otherwise). `embed_distance` is
        always set; `rerank_score` is None when reranking is off.
        """
        # Serialize concurrent callers — see note on _retrieve_lock above.
        with self._retrieve_lock:
            return self._retrieve_impl(
                query, top_k, use_rerank, top_j, max_definition_chunks, doc_id
            )

    def _retrieve_impl(
        self,
        query: str,
        top_k: int,
        use_rerank: bool,
        top_j: int,
        max_definition_chunks: int | None,
        doc_id: str | None,
    ) -> list[dict[str, Any]]:
        log.info(
            "retrieve: query=%r  top_k=%d  use_rerank=%s  top_j=%d  max_def=%s",
            query, top_k, use_rerank, top_j, max_definition_chunks,
        )
        t_total = time.time()
        # When capping defined_term chunks in embed-only mode, over-fetch
        # so there are enough non-definition candidates to fill the slots
        # that the cap leaves open.
        apply_cap = max_definition_chunks is not None and not use_rerank
        n_embed = top_k * 3 if apply_cap else top_k

        t0 = time.time()
        log.info("retrieve: (1/%d) embedding query...", 3 if use_rerank else 2)
        query_emb = self.embedder.encode(
            [query],
            prompt_name=self.query_prompt_name,
            normalize_embeddings=True,
        ).tolist()[0]
        log.info("retrieve: embedded in %.2fs", time.time() - t0)

        t1 = time.time()
        log.info(
            "retrieve: vector search top-%d (doc_id=%s)...",
            n_embed, doc_id,
        )
        where = {"doc_id": doc_id} if doc_id else None
        result = self._collection.query(
            query_embeddings=[query_emb],
            n_results=n_embed,
            where=where,
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        log.info(
            "retrieve: got %d candidates in %.2fs",
            len(documents), time.time() - t1,
        )

        if not documents:
            log.info("retrieve: no candidates, returning empty result")
            return []

        if not use_rerank:
            ranked: list[dict[str, Any]] = []
            n_def_included = 0
            n_def_skipped = 0
            for idx in range(len(documents)):
                meta = metadatas[idx] or {}
                is_def = meta.get("chunk_type") == "defined_term"
                if (
                    apply_cap
                    and is_def
                    and n_def_included >= max_definition_chunks
                ):
                    n_def_skipped += 1
                    continue
                rank = len(ranked) + 1
                score = 1.0 - float(distances[idx])
                log.info(
                    "retrieve:   rank %d: section_id=%s | title=%s | type=%s | score=%.4f",
                    rank,
                    meta.get("section_id", ""),
                    meta.get("section_title", ""),
                    meta.get("chunk_type", ""),
                    score,
                )
                ranked.append(
                    {
                        "rank": rank,
                        "chunk_id": ids[idx],
                        "text": documents[idx],
                        "metadata": metadatas[idx],
                        "embed_distance": distances[idx],
                        "rerank_score": None,
                        "score": score,
                    }
                )
                if is_def:
                    n_def_included += 1
                if len(ranked) >= top_k:
                    break
            if apply_cap:
                log.info(
                    "retrieve: capped defined_term: included=%d skipped=%d (max=%d)",
                    n_def_included, n_def_skipped, max_definition_chunks,
                )
            log.info("retrieve: total %.2fs", time.time() - t_total)
            return ranked

        t2 = time.time()
        log.info(
            "retrieve: reranking %d → top %d...",
            len(documents), top_j,
        )
        rerank_out = self.reranker.rerank(
            query, documents, top_n=top_j
        )
        log.info("retrieve: reranked in %.2fs", time.time() - t2)

        ranked = []
        for rank, item in enumerate(rerank_out, start=1):
            idx = item["index"]
            meta = metadatas[idx] or {}
            score = float(item["relevance_score"])
            log.info(
                "retrieve:   rank %d: section_id=%s | title=%s | score=%.4f",
                rank,
                meta.get("section_id", ""),
                meta.get("section_title", ""),
                score,
            )
            ranked.append(
                {
                    "rank": rank,
                    "chunk_id": ids[idx],
                    "text": documents[idx],
                    "metadata": metadatas[idx],
                    "embed_distance": distances[idx],
                    "rerank_score": score,
                    "score": score,
                }
            )

        log.info("retrieve: total %.2fs", time.time() - t_total)
        return ranked

    def list_documents(self) -> list[dict[str, Any]]:
        all_items = self._collection.get()
        by_doc: dict[str, dict[str, Any]] = {}
        for meta in all_items["metadatas"]:
            doc_id = meta.get("doc_id")
            if doc_id is None:
                continue
            entry = by_doc.setdefault(
                doc_id, {"doc_id": doc_id, "n_chunks": 0, "sections": set()}
            )
            entry["n_chunks"] += 1
            sec = meta.get("section_id")
            if sec:
                entry["sections"].add(sec)
        return [
            {**v, "sections": sorted(v["sections"])} for v in by_doc.values()
        ]

    def delete_document(self, doc_id: str) -> int:
        existing = self._collection.get(where={"doc_id": doc_id})
        n = len(existing["ids"])
        if n:
            self._collection.delete(where={"doc_id": doc_id})
        return n

    @staticmethod
    def _hash_file(path: str) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()[:16]


def _chroma_safe(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    out: dict[str, str | int | float | bool] = {}
    for k, v in meta.items():
        if v is None:
            continue
        out[k] = v
    return out
