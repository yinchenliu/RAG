"""Diagnostic: scan cpi.pdf for multiple TOC headings / document boundaries.

Hypothesis under test: cpi.pdf is actually 3 separate documents (each with
its own Table of Contents) concatenated into one PDF. The current parser
only finds the FIRST TOC, so everything after the first doc is mis-parsed.
"""
import re
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PDF = str(Path(__file__).resolve().parents[1] / "documents" / "cpi.pdf")

doc = fitz.open(PDF)
print(f"cpi.pdf has {len(doc)} pages\n")

# 1. Find every "TABLE OF CONTENTS" / "CONTENTS" heading
toc_re = re.compile(r"\b(TABLE\s+OF\s+CONTENTS|CONTENTS)\b", re.IGNORECASE)
print("=" * 90)
print("Pages containing a TOC heading:")
print("=" * 90)
for i in range(len(doc)):
    text = doc[i].get_text("text")
    for line in text.split("\n"):
        s = line.strip()
        if toc_re.search(s) and len(s) < 60:
            print(f"  PDF page {i+1:>4}  |  {s!r}")
            break

# 2. Look for major document-title markers (agreement openers)
print("\n" + "=" * 90)
print("Candidate document-boundary lines (agreement/indenture openers):")
print("=" * 90)
opener_re = re.compile(
    r"(CREDIT\s+AGREEMENT|INDENTURE|INTERCREDITOR\s+AGREEMENT|"
    r"OFFERING\s+(CIRCULAR|MEMORANDUM)|PRELIMINARY\s+STATEMENTS|"
    r"GUARANTEE|SECURITY\s+AGREEMENT|PURCHASE\s+AGREEMENT)",
    re.IGNORECASE,
)
for i in range(len(doc)):
    text = doc[i].get_text("text")
    for line in text.split("\n"):
        s = line.strip()
        if opener_re.search(s) and 8 < len(s) < 90:
            print(f"  PDF page {i+1:>4}  |  {s!r}")
            break

# 3. Detect printed-page-number resets (a sign of concatenated docs).
print("\n" + "=" * 90)
print("Printed page-number sequence (first standalone number per page):")
print("=" * 90)
pn_re = re.compile(r"^\s*(\d{1,4})\s*$")
seq = []
for i in range(len(doc)):
    text = doc[i].get_text("text")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    found = None
    for c in lines[:3] + lines[-3:]:
        m = pn_re.match(c)
        if m:
            found = int(m.group(1))
            break
    seq.append((i + 1, found))

# Print resets: where printed number drops back toward 1
prev = None
for pdf_pg, printed in seq:
    if printed is not None and prev is not None and printed < prev - 5:
        print(f"  RESET at PDF page {pdf_pg}: printed {prev} -> {printed}")
    if printed is not None:
        prev = printed

doc.close()
