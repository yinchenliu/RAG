"""Direct ChromaDB inspection — no embedding model needed.

Shows common access patterns:
  1. collection.count()                        — how many chunks
  2. collection.peek(limit=N)                  — quick preview
  3. collection.get(limit=N)                   — first N chunks (no ranking)
  4. collection.get(ids=[...])                 — fetch by specific chunk_id
  5. collection.get(where={...})               — filter by metadata
  6. collection.get(where={...}, limit=N)      — filter + paginate
"""
import chromadb

PERSIST_DIR = "RAG/chroma_db"
COLLECTION = "clo_indentures"

client = chromadb.PersistentClient(path=PERSIST_DIR)
col = client.get_collection(name=COLLECTION)

# 1) total count
print(f"Total chunks in collection: {col.count()}\n")

# 2) peek - quick look at first few
print("=" * 90)
print("peek(limit=2) — fast preview")
print("=" * 90)
peek = col.peek(limit=2)
for cid, doc, meta in zip(peek["ids"], peek["documents"], peek["metadatas"]):
    print(f"id={cid}  sec={meta['section_id']}  type={meta['chunk_type']}")
    print(f"  text[:80]: {doc[:80]!r}\n")

# 3) first N chunks (insertion order is NOT guaranteed; use it just to browse)
print("=" * 90)
print("get(limit=5) — first 5 chunks (no semantic ranking, just browse)")
print("=" * 90)
got = col.get(limit=5, include=["documents", "metadatas"])
for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
    print(f"id={cid}  sec={meta['section_id']}  title={meta['section_title']!r}")
    print(f"  chunk_type={meta['chunk_type']}  term={meta.get('term')!r}  pages={meta['page_start']}-{meta['page_end']}")
    print(f"  text[:120]: {doc[:120]!r}\n")

# 4) fetch a specific chunk by id (handy after retrieve() returns chunk_ids)
print("=" * 90)
print("get(ids=[...]) — fetch one known chunk_id")
print("=" * 90)
if got["ids"]:
    target = got["ids"][0]
    one = col.get(ids=[target], include=["documents", "metadatas"])
    print(f"Got id={one['ids'][0]}, metadata={one['metadatas'][0]}\n")

# 5) filter by metadata: e.g. only Section 2.5 chunks
print("=" * 90)
print("get(where={'section_id': '2.5'}) — all chunks from a specific section")
print("=" * 90)
sec25 = col.get(where={"section_id": "2.5"}, include=["documents", "metadatas"])
print(f"Found {len(sec25['ids'])} chunks in Section 2.5")
for cid, meta in zip(sec25["ids"][:3], sec25["metadatas"][:3]):
    print(f"  {cid}  {meta['chunk_type']}  pages={meta['page_start']}-{meta['page_end']}")

# 6) filter by chunk_type + limit
print("\n" + "=" * 90)
print("get(where={'chunk_type': 'defined_term'}, limit=5) — first 5 defined terms")
print("=" * 90)
terms = col.get(
    where={"chunk_type": "defined_term"},
    limit=5,
    include=["documents", "metadatas"],
)
for cid, doc, meta in zip(terms["ids"], terms["documents"], terms["metadatas"]):
    print(f"  term={meta.get('term')!r}")
    print(f"    body[:80]: {doc[:80]!r}")

# 7) compound metadata filter: defined_term AND term starts with "CCC"
print("\n" + "=" * 90)
print('get(where={"$and": [...]}) — compound filter')
print("=" * 90)
# ChromaDB doesn't support string prefix natively. For exact match use $eq;
# for "term contains X" you'd post-filter in Python.
hits = col.get(
    where={"$and": [{"chunk_type": {"$eq": "defined_term"}}, {"section_id": {"$eq": "1.1"}}]},
    include=["metadatas"],
)
print(f"Defined-term chunks in Section 1.1: {len(hits['ids'])}")

# Post-filter example: terms whose text contains "CCC/Caa"
ccc_terms = [
    (cid, meta.get("term"))
    for cid, meta in zip(hits["ids"], hits["metadatas"])
    if meta.get("term") and "CCC/Caa" in meta["term"]
]
print(f"\nDefined terms with 'CCC/Caa' in the term name:")
for cid, term in ccc_terms:
    print(f"  {cid}  {term!r}")
