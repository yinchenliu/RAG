from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Any

import chromadb

from RAG.chunker import Chunk, chunk_sections
from RAG.pre_injection.config import load_config
from RAG.pdf_parser import parse_pdf

log = logging.getLogger(__name__)


class CLOIndentureRAG:
    def __init__(
        self,
        persist_dir: str = "RAG/chroma_db",
        collection_name: str = "clo_indentures",
        embedding_model_name: str = "microsoft/harrier-oss-v1-0.6b",
        query_prompt_name: str = "web_search_query",
        reranker_model_name: str = "jinaai/jina-reranker-v3",
        device: str = "auto",
        max_chunk_tokens: int = 2048,
        chunk_overlap_tokens: int = 128,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.query_prompt_name = query_prompt_name
        self.reranker_model_name = reranker_model_name
        self.device = device
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, embedding_function=None # QTC: for chromadb, is a collection like a table in a relational db (like SQL server)? if so, if I have multiple indentures of multiple deals, do they each saved in separate collection?
        )

        self._embedder = None
        self._reranker = None
        # Serializes retrieve() calls so concurrent tool calls don't race
        # on the Jina reranker's HF fast tokenizer (Rust-backed, NOT
        # thread-safe — raises "Already borrowed" on concurrent use) or
        # on lazy model init.
        self._retrieve_lock = threading.Lock()

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer # QTC: what does sentence_transofrmers lib do? seems that this library can help us create embedding/reranker model object?

            self._embedder = SentenceTransformer(
                self.embedding_model_name,
                device=self._resolve_device(),
                model_kwargs={"dtype": "auto"},
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

        t1 = time.time()
        chunks: list[Chunk] = chunk_sections(
            sections,
            doc_id=doc_id,
            config=config,
            max_tokens=self.max_chunk_tokens,
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

        texts = [c["text"] for c in chunks] #QTC: for the term definitioin, we are not embedding the term? only the body of the term?
        # batch_size=1: attention is O(seq_len^2); with chunks up to
        # max_chunk_tokens (10k), MPS can't fit a batched padded sequence.
        log.info("ingest: embedding %d chunks (batch_size=1)...", len(texts))
        t2 = time.time()
        embeddings = self.embedder.encode(
            texts,
            batch_size=1,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()
        log.info("ingest: embedded in %.2fs", time.time() - t2)

        self._collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[_chroma_safe(c["metadata"]) for c in chunks], #QTC: chroma db can also save a json format of the metadata?
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
            for block in iter(lambda: f.read(1024 * 1024), b""): #QTC: what are we doing here, are we iterating through the entire pdf content to get hashid?
                h.update(block)
        return h.hexdigest()[:16]


def _chroma_safe(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    out: dict[str, str | int | float | bool] = {}
    for k, v in meta.items():
        if v is None:
            continue
        out[k] = v
    return out
