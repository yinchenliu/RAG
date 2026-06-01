"""Quick chunk-size histogram to predict ingest time."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RAG.chunker import chunk_sections, count_tokens
from RAG.pre_injection.config import load_config
from RAG.pdf_parser import parse_pdf

config = load_config("sixth_street_clo_29")
sections = parse_pdf(
    "RAG/Indenture_pdf/4. Sixth Street CLO 29 - Indenture (with Final OC) (Executed).pdf",
    config,
)
chunks = chunk_sections(sections, doc_id="test", config=config, max_tokens=10000)

print(f"Total chunks: {len(chunks)}")
print(f"By chunk_type:")
for t, n in Counter(c["metadata"]["chunk_type"] for c in chunks).most_common():
    print(f"  {t:>15}  {n:>5}")

# Token size histogram
toks = [count_tokens(c["text"]) for c in chunks]
print(f"\nToken size: min={min(toks)} max={max(toks)} mean={sum(toks)/len(toks):.0f}")
buckets = [0, 100, 500, 1000, 2000, 4000, 8000, 10001]
for lo, hi in zip(buckets, buckets[1:]):
    n = sum(1 for t in toks if lo <= t < hi)
    print(f"  [{lo:>5},{hi:>5})  {n:>5}  {'#' * (n * 50 // len(chunks))}")
