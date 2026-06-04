"""End-to-end check: does the coupon-rate query now retrieve real Credit
Agreement content (Applicable Rate / interest margin) instead of the
Intercreditor Agreement exhibit?"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RAG.rag import CLOIndentureRAG

rag = CLOIndentureRAG(persist_dir=str(Path(__file__).resolve().parents[1] / "chroma_db"))

for q in [
    "coupon rate of the initial term loan",
    "Applicable Rate definition term loan margin SOFR",
]:
    print("=" * 90)
    print(f"QUERY: {q}")
    print("=" * 90)
    results = rag.retrieve(q, top_k=8, use_rerank=False, max_definition_chunks=6, doc_id="cpi")
    for r in results[:6]:
        meta = r.get("metadata", {}) or {}
        sid = meta.get("section_id", "")
        title = meta.get("section_title", "")
        ps, pe = meta.get("page_start", ""), meta.get("page_end", "")
        body = " ".join(r["text"].split())[:220]
        print(f"\n[{r['rank']}] {sid} | {title} | pp {ps}-{pe} | score={r['score']:.3f}")
        print(f"    {body}")
    print()
