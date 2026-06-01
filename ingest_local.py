"""Local CPU ingestion run for the CLO indenture RAG.

Ingests the indenture under documents/ into ChromaDB on this machine.

testing/_test_step2_ingest.py points at the MacBook's Indenture_pdf/ layout and
was effectively tuned for the GPU profile (the large harrier model, one chunk at
a time). This run instead relies on the device-aware embedding profile (see
EMBEDDING_PROFILES in rag.py): on a CPU-only box it resolves to the "cpu" profile
(149M ModernBERT, 8192-ctx so chunks embed whole, 512-token chunks, batch 16) —
minutes instead of hours, no text truncated. On the MacBook it resolves to the
"gpu" profile (harrier) automatically — same script, no edits.

Paths are resolved relative to this file, so it can be launched from anywhere:
    python RAG/ingest_local.py
    python -m RAG.ingest_local
"""
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../RAG
sys.path.insert(0, str(HERE.parent))            # so `import RAG` works

from RAG.rag import CLOIndentureRAG

# INFO logging surfaces rag.py's per-stage timings, the resolved
# "profile=... device=... batch_size=..." line, and the no-truncation check,
# so you can see the embedding speed and that nothing was cut off.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

PDF = HERE / "documents" / "CLO 29 - Indenture.pdf"
DOC_ID = "clo"  # → ingest() auto-loads pre_injection/configs/clo.yaml


def main() -> None:
    rag = CLOIndentureRAG(
        persist_dir=str(HERE / "chroma_db"),
        collection_name="clo_indentures",
        device="auto",             # → "cpu" on this box, "mps" on the MacBook
        embedding_profile="auto",  # → "cpu" profile on CPU, "gpu" profile on MPS/CUDA
    )

    print(f"Ingesting {PDF.name!r} as doc_id={DOC_ID!r} (overwrite=True) ...")
    t0 = time.time()
    result = rag.ingest(str(PDF), doc_id=DOC_ID, overwrite=True)
    elapsed = time.time() - t0

    print(f"result : {result}")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"total chunks in collection: {rag._collection.count()}")


if __name__ == "__main__":
    main()
