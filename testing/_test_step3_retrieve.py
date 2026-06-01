"""Step 3+4: retrieve top_k from embedder, then top_j from reranker.

Shows BOTH stages so we can see how the reranker reorders things.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RAG.rag import CLOIndentureRAG

QUERY = "how to calculate the ccc/caa excess adjustment or haircut"
TOP_K_EMBED = 12
TOP_K_RERANK = 5

rag = CLOIndentureRAG(
    persist_dir="RAG/chroma_db",
    collection_name="clo_indentures",
)

print(f"Query: {QUERY!r}\n")

# Stage 1: embedder top-K
print(f"{'=' * 100}")
print(f"STAGE 1 — top {TOP_K_EMBED} from embedder (bi-encoder cosine similarity)")
print(f"{'=' * 100}")
t0 = time.time()
query_emb = rag.embedder.encode(
    [QUERY], prompt_name=rag.query_prompt_name, normalize_embeddings=True
).tolist()[0]
embed_elapsed = time.time() - t0

t0 = time.time()
result = rag._collection.query(query_embeddings=[query_emb], n_results=TOP_K_EMBED)
chroma_elapsed = time.time() - t0

ids = result["ids"][0]
docs = result["documents"][0]
metas = result["metadatas"][0]
dists = result["distances"][0]

print(f"Embed query: {embed_elapsed*1000:.0f}ms   Chroma query: {chroma_elapsed*1000:.0f}ms\n")
print(f"{'rk':>3}  {'chunk_id':>17}  {'sec':>6}  {'chunk_type':>14}  {'dist':>8}  preview")
print(f"{'-' * 100}")
for i, (cid, m, d, dist) in enumerate(zip(ids, metas, docs, dists), start=1):
    term = m.get("term") or ""
    section = m.get("section_id", "?")
    label = f"{m.get('chunk_type', '?')[:14]}"
    preview = (term + " | " + d.replace("\n", " "))[:80] if term else d.replace("\n", " ")[:80]
    print(f"{i:>3}  {cid:>17}  {section:>6}  {label:>14}  {dist:>8.4f}  {preview}")

# Stage 2: reranker
print(f"\n{'=' * 100}")
print(f"STAGE 2 — top {TOP_K_RERANK} from jina-reranker-v3 (rerank the {TOP_K_EMBED} above)")
print(f"{'=' * 100}")
t0 = time.time()
rerank_out = rag.reranker.rerank(QUERY, docs, top_n=TOP_K_RERANK)
rerank_elapsed = time.time() - t0
print(f"Rerank: {rerank_elapsed*1000:.0f}ms\n")

print(f"{'rk':>3}  {'embed_rk':>9}  {'chunk_id':>17}  {'sec':>6}  {'chunk_type':>14}  {'score':>8}  preview")
print(f"{'-' * 110}")
for new_rank, item in enumerate(rerank_out, start=1):
    idx = item["index"]
    cid = ids[idx]
    m = metas[idx]
    d = docs[idx]
    term = m.get("term") or ""
    section = m.get("section_id", "?")
    label = f"{m.get('chunk_type', '?')[:14]}"
    preview = (term + " | " + d.replace("\n", " "))[:80] if term else d.replace("\n", " ")[:80]
    print(
        f"{new_rank:>3}  {idx+1:>9}  {cid:>17}  {section:>6}  {label:>14}  "
        f"{float(item['relevance_score']):>8.4f}  {preview}"
    )

# Show full text of top-1 reranked
print(f"\n{'=' * 100}")
print(f"TOP-1 RERANKED chunk full text:")
print(f"{'=' * 100}")
top_idx = rerank_out[0]["index"]
print(f"section_id: {metas[top_idx].get('section_id')}")
print(f"section_title: {metas[top_idx].get('section_title')}")
print(f"chunk_type: {metas[top_idx].get('chunk_type')}")
print(f"pages: {metas[top_idx].get('page_start')}-{metas[top_idx].get('page_end')}")
if metas[top_idx].get("term"):
    print(f"term: {metas[top_idx]['term']!r}")
print(f"\n--- body ({len(docs[top_idx])} chars) ---")
print(docs[top_idx][:3000])
if len(docs[top_idx]) > 3000:
    print(f"\n... [truncated, {len(docs[top_idx]) - 3000} more chars]")
