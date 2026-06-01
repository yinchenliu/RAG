"""Step 2: embed all chunks and ingest into ChromaDB.

First run will download harrier-oss-v1-0.6b (~1.2GB).
"""
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RAG.rag import CLOIndentureRAG

# One entry per document to ingest. The `doc_id` is also used to look up the
# per-document config at `RAG/configs/<doc_id>.yaml`; if no such file exists,
# the ingest falls back to `_defaults`.
DOCS = [
    {
        "pdf": "RAG/Indenture_pdf/4. Sixth Street CLO 29 - Indenture (with Final OC) (Executed).pdf",
        "doc_id": "sixth_street_clo_29",
    },
    {
        "pdf": "RAG/Indenture_pdf/Sample Credit Agreement.pdf",
        "doc_id": "cpi_holdco_credit_agreement",
    },
]

# Legacy hash-based doc_id used before we switched to human-readable IDs.
# Delete it on this run so we don't end up with two copies of the same
# indenture under different doc_ids.
LEGACY_HASH_IDS = ["bd70ffeb2c9b8593"]

print("Instantiating CLOIndentureRAG (this triggers chroma client init)...")
rag = CLOIndentureRAG(
    persist_dir="RAG/chroma_db",
    collection_name="clo_indentures",
)

for legacy_id in LEGACY_HASH_IDS:
    n_deleted = rag.delete_document(legacy_id)
    if n_deleted:
        print(f"Deleted {n_deleted} legacy chunks under doc_id={legacy_id!r}")

for spec in DOCS:
    print(f"\n{'=' * 80}")
    print(f"Ingesting doc_id={spec['doc_id']!r}")
    print(f"  pdf={spec['pdf']}")
    print(f"{'=' * 80}")
    t0 = time.time()
    result = rag.ingest(
        spec["pdf"],
        doc_id=spec["doc_id"],
        overwrite=True,
    )
    elapsed = time.time() - t0
    print(f"  result: {result}")
    print(f"  elapsed: {elapsed:.1f}s")

print(f"\n{'=' * 80}")
print("Collection summary")
print(f"{'=' * 80}")
print(f"Total chunks in collection: {rag._collection.count()}")
for doc_info in rag.list_documents():
    print(
        f"  doc_id={doc_info['doc_id']!r}  "
        f"n_chunks={doc_info['n_chunks']}  "
        f"n_sections={len(doc_info['sections'])}"
    )

all_items = rag._collection.get()
type_counts = Counter(m["chunk_type"] for m in all_items["metadatas"])
print("\nGlobal chunk_type counts:")
for ct, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {ct:>15}  {count:>5}")
