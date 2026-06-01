"""Dump raw TOC text to diagnose why Chapter IV / XII aren't matching, and
to find the schedule heading format."""
import fitz

doc = fitz.open(
    "RAG/Indenture_pdf/4. Sixth Street CLO 29 - Indenture (with Final OC) (Executed).pdf"
)

print("=" * 90)
print("Raw TOC text (pages 0-7):")
print("=" * 90)
for i in range(min(8, len(doc))):
    print(f"\n--- PDF page index {i} ---")
    print(doc[i].get_text("text"))

print("\n" + "=" * 90)
print("Search the whole doc for chapter / article / schedule / exhibit headings")
print("=" * 90)
import re

for i in range(len(doc)):
    text = doc[i].get_text("text")
    for line in text.split("\n"):
        s = line.strip()
        if re.match(r"^(ARTICLE|CHAPTER|SCHEDULE|EXHIBIT|APPENDIX|ANNEX)\b", s, re.IGNORECASE):
            print(f"page {i:>3}  |  {s!r}")

doc.close()
