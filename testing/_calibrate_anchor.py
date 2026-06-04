"""Calibrate the fix-#2 anchor-match metric: what fraction of parsed
sections actually contain their own section_id/title on the page the
parser assigned as page_start? Run against the CURRENT parse (broken
offset) to get the 'bad' number for threshold calibration."""
import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fitz
from RAG.pdf_parser import parse_pdf
from RAG.pre_injection.config import load_config

PDF = str(Path(__file__).resolve().parents[1] / "documents" / "cpi.pdf")


def anchor_match_rate(doc, sections) -> float:
    non_empty = [s for s in sections if s["text"].strip()]
    if not non_empty:
        return 0.0
    matched = 0
    for s in non_empty:
        ps = s["page_start"] - 1
        texts = [doc[i].get_text("text") for i in (ps - 1, ps, ps + 1) if 0 <= i < len(doc)]
        page_text = "\n".join(texts)
        sid = (s["section_id"] or "").strip()
        title = (s["section_title"] or "").strip()
        ok = False
        if sid and re.search(r"(?:Section\s+)?" + re.escape(sid) + r"\b", page_text):
            ok = True
        elif len(title) >= 6 and title[:40] in page_text:
            ok = True
        if ok:
            matched += 1
    return matched / len(non_empty)


doc = fitz.open(PDF)
sections = parse_pdf(PDF, load_config("cpi"))
rate = anchor_match_rate(doc, sections)
print(f"sections={len(sections)}  anchor_match_rate={rate:.2%}")
doc.close()
