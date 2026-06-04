"""Confirm the page-offset root cause and locate the real term-loan content."""
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

print("=" * 90)
print("First PDF page whose first/last 3 lines contain a standalone '1':")
print("=" * 90)
for i in range(len(doc)):
    lines = [l.strip() for l in doc[i].get_text("text").split("\n") if l.strip()]
    cand = lines[:3] + lines[-3:]
    for c in cand:
        m = pn_re.match(c)
        if m and int(m.group(1)) == 1:
            print(f"  -> PDF page index {i} (PDF page {i+1}); edge lines: {cand}")
            break
    else:
        continue
    break

print("\n" + "=" * 90)
print("Where does the Credit Agreement's Section 1.01 body actually start?")
print("=" * 90)
hdr = re.compile(r"Section\s+1\.01\b")
for i in range(len(doc)):
    if hdr.search(doc[i].get_text("text")):
        print(f"  'Section 1.01' appears on PDF page {i+1}")

print("\n" + "=" * 90)
print("Locate the term-loan rate content (Applicable Rate / Margin / Interest):")
print("=" * 90)
needles = ["Applicable Rate", "Applicable Margin", "Initial Term Loan", "Term B Loan"]
for needle in needles:
    pages = []
    for i in range(len(doc)):
        if needle in doc[i].get_text("text"):
            pages.append(i + 1)
    head = pages[:8]
    print(f"  {needle!r}: {len(pages)} pages -> first few: {head}")

doc.close()
