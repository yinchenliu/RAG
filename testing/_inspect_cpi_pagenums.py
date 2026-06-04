"""Map pdf_idx -> detected printed page number, and dump edge lines for the
Credit Agreement's early body pages, to understand why offset detection
missed the CA's own 'page 1'."""
import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fitz

PDF = str(Path(__file__).resolve().parents[1] / "documents" / "cpi.pdf")
doc = fitz.open(PDF)
pn_re = re.compile(r"^\s*(\d{1,4})\s*$")


def detect(i, first_n=3, last_n=3):
    lines = [l.strip() for l in doc[i].get_text("text").split("\n") if l.strip()]
    for c in lines[:first_n] + lines[-last_n:]:
        m = pn_re.match(c)
        if m:
            return int(m.group(1))
    return None


print("Detected printed page number per PDF page (first 30 pages):")
for i in range(min(30, len(doc))):
    print(f"  PDF page {i+1:>3}: detected={detect(i)}")

print("\nEdge lines (first 3 + last 3 non-empty) for PDF pages 5-10:")
for i in range(4, 10):
    lines = [l.strip() for l in doc[i].get_text("text").split("\n") if l.strip()]
    print(f"\n  --- PDF page {i+1} ---")
    print(f"    first3: {lines[:3]}")
    print(f"    last3 : {lines[-3:]}")

doc.close()
