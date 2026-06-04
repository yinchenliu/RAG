"""Run the real parser on cpi.pdf with its current config and dump every
parsed section's page range + title, plus the TOC page text and the
detected page offset. This shows whether sections are mapped to the wrong
PDF pages (the symptom: section 1.01 'Defined Terms' pointing at pp 309-400,
which is the Intercreditor Agreement exhibit, not the Credit Agreement defs).
"""
import io
import sys
from pathlib import Path

# Force UTF-8 so curly quotes / bullets don't crash on Windows cp1252.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fitz
from RAG.pdf_parser import _parse_via_toc, _detect_page_offset, parse_pdf
from RAG.pre_injection.config import load_config

PDF = str(Path(__file__).resolve().parents[1] / "documents" / "cpi.pdf")
config = load_config("cpi")

print("=" * 90)
print("TOC page (PDF page index 1) raw text:")
print("=" * 90)
doc = fitz.open(PDF)
print(doc[1].get_text("text")[:3000])

print("\n" + "=" * 90)
print("Parsed sections (section_id | pp start-end | title):")
print("=" * 90)
sections = parse_pdf(PDF, config)
print(f"\nTotal sections: {len(sections)}\n")
for s in sections:
    print(f"  {s['section_id']:>8}  pp {s['page_start']:>3}-{s['page_end']:<3}  {s['section_title'][:60]}")

# What offset did detection pick for the first TOC entry?
print("\n" + "=" * 90)
print("Page-offset detection diagnostics:")
print("=" * 90)
toc_sections = _parse_via_toc(doc, config)
print(f"  _parse_via_toc returned {len(toc_sections)} sections")
if toc_sections:
    first = toc_sections[0]
    print(f"  first parsed section: {first['section_id']} -> pp {first['page_start']}-{first['page_end']}")

doc.close()
