"""Step 1: verify TOC parsing produces a clean section list."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RAG.pre_injection.config import load_config
from RAG.pdf_parser import parse_pdf

PDF = "RAG/Indenture_pdf/4. Sixth Street CLO 29 - Indenture (with Final OC) (Executed).pdf"

config = load_config("sixth_street_clo_29")
sections = parse_pdf(PDF, config)

print(f"\n{'=' * 90}")
print(f"Parsed {len(sections)} sections")
print(f"{'=' * 90}\n")

print(f"{'idx':>4}  {'chap':>5}  {'sec_id':>7}  {'pages':>11}  {'len':>6}  title")
print(f"{'-' * 90}")
for i, s in enumerate(sections):
    pages = f"{s['page_start']}-{s['page_end']}"
    body_len = len(s["text"])
    title = s["section_title"][:50]
    print(
        f"{i:>4}  {str(s['chapter'] or '')[:5]:>5}  {s['section_id']:>7}  "
        f"{pages:>11}  {body_len:>6}  {title}"
    )

print(f"\n{'=' * 90}")
print("Quick sanity checks:")
print(f"{'=' * 90}")

if sections:
    s0 = sections[0]
    print(f"\nSection 0: id={s0['section_id']!r} title={s0['section_title']!r}")
    print(f"  pages: {s0['page_start']}-{s0['page_end']}  body chars: {len(s0['text'])}")
    print(f"  body preview (first 400 chars):\n    {s0['text'][:400]!r}")

    defs = [
        s
        for s in sections
        if s["section_id"] in config.definitions.section_id_hints
        or any(
            s["section_title"].lower().startswith(p.lower())
            for p in config.definitions.title_prefix_hints
        )
    ]
    if defs:
        d = defs[0]
        print(f"\nDefinitions section: id={d['section_id']!r} title={d['section_title']!r}")
        print(f"  pages: {d['page_start']}-{d['page_end']}  body chars: {len(d['text'])}")
        print(f"  body preview (first 600 chars):\n    {d['text'][:600]!r}")
    else:
        print("\nNo Definitions section detected by configured hints")
