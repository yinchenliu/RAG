# CLO Indenture RAG

A retrieval-augmented-generation pipeline for **CLO indentures, offering circulars, and credit agreements** — long, densely-structured legal PDFs. It parses a PDF into its Article/Section hierarchy, chunks it (with special handling for the ~90-page definitions block), embeds the chunks into a persistent [ChromaDB](https://www.trychroma.com/) collection, and exposes retrieval through a LangGraph chat agent.

The parser and chunker are **document-independent engines**: all the document-specific patterns (TOC format, section-header regexes, definition delimiters) live in external YAML configs. An agentic loop generates a new config for each PDF, so onboarding a new document does not require code changes.

## Pipeline at a glance

```
PDF ─▶ configure ─▶ <doc_id>.yaml ─▶ parse_pdf ─▶ chunk_sections ─▶ embed ─▶ ChromaDB
                       (config)        (sections)     (chunks)                    │
                                                                                  ▼
                                                  LangGraph agent  ◀── retrieve (+ optional rerank)
```

1. **Configure** — `pre_injection/configure.py` reads the first ~20 pages, runs an agentic propose → validate → smoke-test → critique loop (Gemini), and writes a validated `pre_injection/configs/<doc_id>.yaml`.
2. **Parse** — `pdf_parser.py` runs TOC-first parsing (with a linewalk fallback) to produce a clean list of sections, stripping running headers/footers and detecting printed-page offsets.
3. **Chunk** — `chunker.py` splits sections into token-bounded chunks, handling the definitions section term-by-term.
4. **Ingest / retrieve** — `rag.py` (`CLOIndentureRAG`) embeds chunks, stores them in ChromaDB, and serves vector retrieval with an optional cross-encoder rerank stage.
5. **Chat** — `RAG_graph.py` is a LangGraph agent that exposes `list_indentures` and `search_clo_indenture` tools over the store.

## Project structure

| Path | Purpose |
|------|---------|
| [rag.py](rag.py) | `CLOIndentureRAG` — ingest, retrieve, list/delete documents; wraps ChromaDB + embedder + reranker |
| [pdf_parser.py](pdf_parser.py) | `parse_pdf(pdf_path, config)` — config-driven TOC/section parser (PyMuPDF) |
| [chunker.py](chunker.py) | `chunk_sections(...)` — token-bounded chunking with definitions-aware splitting |
| [RAG_graph.py](RAG_graph.py) | LangGraph chat agent with retrieval tools (interactive CLI) |
| [pre_injection/config.py](pre_injection/config.py) | Pydantic config schema + `load_config()` (validates & precompiles regexes) |
| [pre_injection/configure.py](pre_injection/configure.py) | CLI entry point for the agentic config generator |
| [pre_injection/configure_graph.py](pre_injection/configure_graph.py) | The propose/validate/smoke-test/critique LangGraph loop |
| [pre_injection/configs/](pre_injection/configs/) | `_defaults.yaml` baseline + one `<doc_id>.yaml` per document |
| [testing/](testing/) | Step-by-step verification scripts (parse → ingest → retrieve) and inspectors |
| [plan.md](plan.md) | Design notes for the config-externalization refactor |

> **Note on imports & working directory.** The code is a package imported as `RAG`, and all PDF paths are written relative to the package's parent (e.g. `RAG/Indenture_pdf/...`). Run every command below from the directory **above** this one (the parent of `RAG/`).

## Setup

Requires **Python 3.11+**.

```bash
# from the parent directory of RAG/
python -m venv RAG/.venv
source RAG/.venv/bin/activate
pip install -r RAG/requirements.txt
```

The parse/chunk/ingest/retrieve path needs only the core deps (PyMuPDF, ChromaDB, sentence-transformers, transformers, torch, tiktoken, langchain-text-splitters). The agentic configurator and chat agent additionally use `langgraph`, `langchain-google-genai`, and `dotenv`.

### Environment variables

Create a `.env` file in this directory (it is gitignored):

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...          # or GOOGLE_API_KEY — used by the configurator & chat agent
```

- `GEMINI_API_KEY` / `GOOGLE_API_KEY` — required to run the configurator (`pre_injection/configure.py`) and the chat agent (`RAG_graph.py`), which call Gemini.
- `ANTHROPIC_API_KEY` — for LLM-assisted extraction steps.

The embedding model and reranker (`jinaai/jina-reranker-v3`) are downloaded from Hugging Face on first use.

The embedding model is chosen by a **device-aware profile** (the `EmbeddingProfile` entries in `EMBEDDING_PROFILES`, in `rag.py`), resolved automatically from the detected device so the same code runs on both machines:

| Device | Profile | Model | Chunk tokens | Model window | Notes |
| --- | --- | --- | --- | --- | --- |
| MPS / CUDA (Apple MacBook) | `gpu` | `microsoft/harrier-oss-v1-0.6b` (~1.2 GB) | 2048 | 8192 | High quality; fast on GPU/MPS, painfully slow on CPU |
| CPU (Windows PC) | `cpu` | `Alibaba-NLP/gte-modernbert-base` (~600 MB) | 512 | 8192 | ~4× smaller + smaller chunks → minutes, not hours; long window so nothing is truncated |

`max_chunk_tokens` is counted in cl100k tokens; the model window is the model's own tokens (dense legal text retokenizes to ~1.3–1.8× more), so chunk budgets stay well under the window. `ingest()` also runs a live check and warns if any chunk would be truncated.

Each profile bundles the model, its query prompt, the chunk size, the batch size, and the window limit — these must stay consistent, since vectors from different models aren't comparable. Override with `CLOIndentureRAG(embedding_profile="gpu")` or per-field args (`embedding_model_name=`, `max_chunk_tokens=`, …). Because the vector dimension differs per model, **each machine keeps its own `chroma_db`** (it's gitignored); re-ingest after switching profiles.

## Usage

### 1. Generate a config for a new PDF

```bash
python -m RAG.pre_injection.configure "RAG/Indenture_pdf/My New Indenture.pdf" --doc-id my_deal
```

The loop writes `RAG/pre_injection/configs/my_deal.yaml` on success, or `my_deal.draft.yaml` for hand-fixup if it fails to converge within `--max-iter` (default 5). Without a per-doc config, ingest falls back to `_defaults.yaml`.

### 2. Ingest a document

```python
from RAG.rag import CLOIndentureRAG

rag = CLOIndentureRAG()
rag.ingest(
    "RAG/Indenture_pdf/My New Indenture.pdf",
    doc_id="my_deal",           # also used to find configs/my_deal.yaml
    overwrite=False,
)
```

Or ingest the local indenture (`documents/CLO 29 - Indenture.pdf`) with the auto-profile run — picks the `cpu` profile on a Windows PC, the `gpu` profile on the MacBook, no edits needed:

```bash
python RAG/ingest_local.py
```

Or use the step scripts to ingest the sample corpus end to end:

```bash
python RAG/testing/_test_step1_parse.py      # inspect parsed sections
python RAG/testing/_test_step2_ingest.py     # embed + ingest into ChromaDB
python RAG/testing/_test_step3_retrieve.py   # retrieve + rerank a sample query
```

### 3. Retrieve

```python
results = rag.retrieve(
    query="how to calculate the CCC/Caa excess adjustment",
    top_k=12,
    use_rerank=True,   # cross-encoder rerank → top_j
    top_j=5,
    doc_id="my_deal",  # scope to one deal — defined terms are deal-specific
)
```

`use_rerank=False` returns the raw top-`top_k` embedding hits (with an optional `max_definition_chunks` cap so short defined-term chunks don't crowd out body sections).

### 4. Chat

```bash
python -m RAG.RAG_graph
```

An interactive REPL. The agent calls `list_indentures` to resolve which deal you mean, then `search_clo_indenture(doc_id, query)` to answer, citing `section_id`s. Type `quit` or `exit` to leave.

## How configuration works

`load_config(doc_id)` resolves `configs/<doc_id>.yaml`, follows its `extends: _defaults` chain, deep-merges over `_defaults.yaml`, validates against the Pydantic schema in [pre_injection/config.py](pre_injection/config.py), and precompiles every regex. The schema groups settings into `toc`, `body`, `page_numbering`, `header_footer_strip`, and `definitions` — see [plan.md](plan.md) for the full design rationale.

Because regexes are validated and capturing-group counts checked at load time, a malformed config fails fast with a field-named error rather than crashing mid-parse — which is what makes the agentic loop's "validate" step reliable.

## Notes

- **One ChromaDB collection holds all deals**; retrieval is scoped per deal via the `doc_id` metadata filter. Always pass `doc_id` when querying so defined terms (e.g. "Closing Date", "Trustee") don't leak across documents.
- `retrieve()` is serialized with a lock — the Jina reranker's Rust tokenizer is not thread-safe under concurrent tool calls.
- The Chroma store, virtualenv, `__pycache__`, source PDFs, and `.env` are gitignored; the vector store is regenerated by re-ingesting.
