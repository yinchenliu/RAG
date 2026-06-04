from __future__ import annotations

import logging
import re
from collections import Counter
from typing import TypedDict

import fitz

from RAG.pre_injection.config import HeaderFooterStripConfig, ParsedConfig

log = logging.getLogger(__name__)


class Section(TypedDict):
    chapter: str | None
    section_id: str
    section_title: str
    text: str
    page_start: int
    page_end: int


def parse_pdf(pdf_path: str, config: ParsedConfig) -> list[Section]:
    """Parse a legal-document PDF into a list of Section dicts.

    Unified algorithm: TOC-first; if it yields fewer than
    `config.toc.min_sections_to_accept` sections, fall back to a line-walk.
    All document-format assumptions (regex patterns, thresholds, body-start
    markers) come from `config` — the parser is document-independt.
    """
    doc = fitz.open(pdf_path)
    try:
        sections = _parse_via_toc(doc, config)
        if len(sections) >= config.toc.min_sections_to_accept:
            return sections
        log.info(
            "TOC parsing produced %d sections (< %d); falling back to line-walk",
            len(sections), config.toc.min_sections_to_accept,
        )
        return _parse_via_linewalk(doc, config)
    finally:
        doc.close()


def _parse_via_toc(doc: fitz.Document, config: ParsedConfig) -> list[Section]:
    toc = config.toc
    search_limit = min(len(doc), toc.search_pages)
    page_texts = [doc[i].get_text("text") for i in range(search_limit)]

    toc_start = None
    for i, text in enumerate(page_texts):
        if toc.heading_re.search(text):
            toc_start = i
            break
    if toc_start is None:
        return []

    entries: list[tuple[str | None, str, str, int]] = []
    current_chapter: str | None = None
    blank_streak = 0

    for page_idx in range(toc_start, search_limit):
        page_text = doc[page_idx].get_text("text")
        page_had_entry = False

        raw_lines = page_text.split("\n")
        lines = [l.rstrip() for l in raw_lines if l.strip()]

        i = 0
        while i < len(lines):
            line = lines[i]

            chapter_match = toc.chapter_re.match(line)
            consumed_continuation = False
            if chapter_match is None and i + 1 < len(lines):
                merged = line + " " + lines[i + 1]
                chapter_match = toc.chapter_re.match(merged)
                if chapter_match is not None:
                    consumed_continuation = True
            if chapter_match:
                current_chapter = chapter_match.group(1)
                i += 2 if consumed_continuation else 1
                page_had_entry = True
                continue

            entry = _match_toc_entry(line, toc.entry_res)
            consumed_entry_extra = 0
            if entry is None:
                for lookahead in (1, 2):
                    if i + lookahead >= len(lines):
                        break
                    merged = " ".join(lines[i : i + lookahead + 1])
                    entry = _match_toc_entry(merged, toc.entry_res)
                    if entry is not None:
                        consumed_entry_extra = lookahead
                        break

            if entry is not None:
                section_id, title, page_num = entry
                entries.append((current_chapter, section_id, title.strip(), page_num))
                page_had_entry = True

            i += 1 + consumed_entry_extra

        if page_had_entry:
            blank_streak = 0
        else:
            blank_streak += 1
            if blank_streak >= toc.blank_page_streak_stop:
                break

    if not entries:
        return []

    offset = _detect_page_offset(doc, entries[0][3], config)
    return _build_sections_from_toc(doc, entries, offset, config)


def _match_toc_entry(
    line: str, patterns: list[re.Pattern[str]]
) -> tuple[str, str, int] | None:
    for pattern in patterns:
        m = pattern.match(line)
        if m:
            try:
                return m.group(1), m.group(2), int(m.group(3))
            except ValueError:
                continue
    return None


_MIN_OFFSET_RUN = 3


def _detect_page_offset(
    doc: fitz.Document, first_printed: int, config: ParsedConfig
) -> int:
    """Find the offset such that pdf_index = printed_num + offset.

    Strategy: read each page's standalone printed number (from its first /
    last few lines), then find the longest run of pages whose numbers
    advance in lockstep with the PDF index — i.e. a constant
    `pdf_index - printed` offset. That run is the document's main body
    numbering sequence, and its offset is what we want.

    This is robust to two things the old "find the page that shows
    `first_printed`" approach got wrong:
      * the first body page often suppresses its own page number, so
        `first_printed` is never actually printed anywhere, and
      * composite PDFs (a credit agreement followed by exhibits) restart
        numbering at 1 several times, so a lone "1" can belong to an
        exhibit hundreds of pages in — which is exactly what shifted every
        cpi.pdf section by +307.
    `first_printed` is kept only as a single-page fallback anchor.
    """
    pn = config.page_numbering

    # 1. Detect at most one standalone printed number per page (edges only).
    detected: list[tuple[int, int]] = []  # (pdf_idx, printed_num)
    for pdf_idx in range(len(doc)):
        lines = [
            l.strip() for l in doc[pdf_idx].get_text("text").split("\n") if l.strip()
        ]
        for c in lines[: pn.search_first_n_lines] + lines[-pn.search_last_n_lines :]:
            m = pn.printed_page_re.match(c)
            if m and m.group(1).isdigit():
                detected.append((pdf_idx, int(m.group(1))))
                break

    # 2. Group consecutive detections sharing a constant offset into runs.
    #    A page with no detectable number simply drops out of `detected`
    #    without breaking the run, since the surviving neighbours still
    #    share the same offset. Prefer the longest run; break ties toward
    #    the earliest (the body sequence precedes any trailing exhibits).
    best_offset: int | None = None
    best_len = 0
    best_start = len(doc)
    run_offset: int | None = None
    run_len = 0
    run_start = 0
    for pdf_idx, printed in detected:
        offset = pdf_idx - printed
        if offset == run_offset:
            run_len += 1
        else:
            run_offset, run_len, run_start = offset, 1, pdf_idx
        if run_len > best_len or (run_len == best_len and run_start < best_start):
            best_offset, best_len, best_start = run_offset, run_len, run_start

    if best_offset is not None and best_len >= _MIN_OFFSET_RUN:
        log.info(
            "page offset = %d (consistent run of %d numbered pages from PDF page %d)",
            best_offset, best_len, best_start + 1,
        )
        return best_offset

    # 3. Fallback: the old single-page anchor on `first_printed`.
    for pdf_idx, printed in detected:
        if printed == first_printed:
            log.info(
                "page offset = %d (fallback: first page showing printed %d)",
                pdf_idx - first_printed, first_printed,
            )
            return pdf_idx - first_printed
    log.warning(
        "Could not detect page-number offset; defaulting to 0 (section page ranges may be shifted)"
    )
    return 0


def _build_sections_from_toc(
    doc: fitz.Document,
    entries: list[tuple[str | None, str, str, int]],
    offset: int,
    config: ParsedConfig,
) -> list[Section]:
    n_pages = len(doc)
    sections: list[Section] = []
    dropped: list[str] = []

    for i, (chapter, section_id, title, printed_page) in enumerate(entries):
        pdf_start = max(0, printed_page + offset)
        # Drop TOC entries whose printed page lies past the doc (happens when
        # the PDF is partial — TOC references pages that aren't rendered).
        if pdf_start >= n_pages:
            dropped.append(section_id)
            continue
        if i + 1 < len(entries):
            next_section_id: str | None = entries[i + 1][1]
            next_pdf_start = max(0, entries[i + 1][3] + offset)
            pdf_end = max(pdf_start, next_pdf_start)
        else:
            next_section_id = None
            cutoff = _find_schedule_cutoff(doc, pdf_start, config)
            pdf_end = cutoff if cutoff is not None else n_pages - 1
        pdf_end = min(pdf_end, n_pages - 1)

        body = _extract_pages(doc, pdf_start, pdf_end)
        body = _trim_to_section_boundaries(body, section_id, next_section_id)
        if config.header_footer_strip.enable:
            body = _strip_running_headers_footers(
                doc, body, pdf_start, pdf_end, config.header_footer_strip
            )

        sections.append(
            Section(
                chapter=chapter,
                section_id=section_id,
                section_title=title,
                text=body,
                page_start=pdf_start + 1,
                page_end=pdf_end + 1,
            )
        )

    if dropped:
        log.warning(
            "TOC referenced %d sections past doc length (%d pages); dropped: %s%s",
            len(dropped), n_pages,
            ", ".join(dropped[:5]),
            f", ... ({len(dropped) - 5} more)" if len(dropped) > 5 else "",
        )

    return sections


def _trim_to_section_boundaries(
    raw_text: str,
    section_id: str,
    next_section_id: str | None,
) -> str:
    """Trim an extracted page-range body to a single section's content.

    Two adjacent TOC entries can share a printed page, so naive whole-page
    extraction puts the neighbor's content into this section. Anchors on the
    body header lines for this section and the next.
    """
    own_header_re = _body_header_re(section_id)
    m = own_header_re.search(raw_text)
    if m:
        raw_text = raw_text[m.end():]

    if next_section_id:
        next_header_re = _body_header_re(next_section_id)
        m = next_header_re.search(raw_text)
        if m:
            raw_text = raw_text[: m.start()]

    return raw_text


def _body_header_re(section_id: str) -> re.Pattern[str]:
    return re.compile(
        r"^\s*(?:Section\s+)?" + re.escape(section_id) + r"\.?(?:\s+\S.*)?\s*$",
        re.MULTILINE,
    )


def _find_schedule_cutoff(
    doc: fitz.Document, start_idx: int, config: ParsedConfig
) -> int | None:
    sched_re = config.body.schedule_cutoff_re
    for pdf_idx in range(start_idx + 1, len(doc)):
        text = doc[pdf_idx].get_text("text")
        if sched_re.search(text):
            return max(start_idx, pdf_idx - 1)
    return None


def _extract_pages(doc: fitz.Document, start: int, end: int) -> str:
    return "\n".join(doc[i].get_text("text") for i in range(start, end + 1))


def _strip_running_headers_footers(
    doc: fitz.Document,
    body: str,
    start: int,
    end: int,
    cfg: HeaderFooterStripConfig,
) -> str:
    if end - start < cfg.min_span_pages:
        return body

    counter: Counter[str] = Counter()
    span = end - start + 1
    for i in range(start, end + 1):
        page_text = doc[i].get_text("text")
        page_lines = [l.strip() for l in page_text.split("\n") if l.strip()]
        edges = page_lines[:2] + page_lines[-2:]
        for line in edges:
            if cfg.min_line_length <= len(line) <= cfg.max_line_length:
                counter[line] += 1

    threshold = max(2, int(span * cfg.min_repetition_fraction))
    repeated = {line for line, count in counter.items() if count >= threshold}
    if not repeated:
        return body

    kept = [l for l in body.split("\n") if l.strip() not in repeated]
    return "\n".join(kept)


def _parse_via_linewalk(doc: fitz.Document, config: ParsedConfig) -> list[Section]:
    body_cfg = config.body
    start_page = _find_linewalk_start(doc, config)
    log.info("linewalk: starting at PDF page %d", start_page + 1)

    sections: list[Section] = []
    current: dict | None = None
    current_chapter: str | None = None
    pending_section_id: str | None = None
    pending_section_page: int | None = None

    sched_re = body_cfg.schedule_cutoff_re
    page_num_re = config.page_numbering.printed_page_re
    skip_res = body_cfg.line_skip_res

    for page_idx in range(start_page, len(doc)):
        page_text = doc[page_idx].get_text("text")

        if sched_re.search(page_text):
            break

        for raw_line in page_text.split("\n"):
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                if current is not None and pending_section_id is None:
                    current["text_lines"].append("")
                continue

            if page_num_re.match(stripped):
                continue
            if any(p.match(stripped) for p in skip_res):
                continue

            chapter_match = body_cfg.chapter_re.match(stripped)
            if not chapter_match:
                chapter_match = body_cfg.chapter_allcaps_re.match(stripped)
            if chapter_match:
                current_chapter = chapter_match.group(1)
                continue

            if pending_section_id is not None:
                # Line after a bare "Section X.Y" header. Some doc families
                # pack the first sentence of the body onto this line,
                # separated from the title by `.<2+spaces>`.
                title, body_first_line = _split_title_and_body(stripped)
                if current is not None:
                    current["page_end"] = max(
                        current["page_start"], (pending_section_page or page_idx) + 1
                    )
                    sections.append(_finalize_section(current))
                section_start_page = (pending_section_page or page_idx) + 1
                current = {
                    "chapter": current_chapter,
                    "section_id": pending_section_id,
                    "section_title": title,
                    "page_start": section_start_page,
                    "page_end": page_idx + 1,
                    "text_lines": [body_first_line] if body_first_line else [],
                }
                pending_section_id = None
                pending_section_page = None
                continue

            id_only_match = body_cfg.section_id_only_re.match(stripped)
            if id_only_match:
                pending_section_id = id_only_match.group(1)
                pending_section_page = page_idx
                continue

            section_match = body_cfg.section_header_re.match(stripped)
            if section_match:
                if current is not None:
                    current["page_end"] = max(current["page_start"], page_idx + 1)
                    sections.append(_finalize_section(current))
                current = {
                    "chapter": current_chapter,
                    "section_id": section_match.group(1),
                    "section_title": section_match.group(2).strip().rstrip("."),
                    "page_start": page_idx + 1,
                    "page_end": page_idx + 1,
                    "text_lines": [],
                }
                continue

            if current is not None:
                current["text_lines"].append(line)
                current["page_end"] = page_idx + 1

    if current is not None:
        if current["page_end"] < current["page_start"]:
            current["page_end"] = current["page_start"]
        sections.append(_finalize_section(current))

    return sections


def _find_linewalk_start(doc: fitz.Document, config: ParsedConfig) -> int:
    """Find the PDF page where body parsing should begin.

    If `config.body.body_start_markers` is non-empty, scan forward for the
    first page containing any marker. Otherwise default to skipping
    `config.body.page_skip_after_toc` pages of front matter.
    """
    markers = config.body.body_start_markers
    if not markers:
        return min(config.body.page_skip_after_toc, len(doc) - 1)

    scan_limit = min(len(doc), max(config.toc.search_pages, 10) * 2)
    for pdf_idx in range(1, scan_limit):
        text = doc[pdf_idx].get_text("text")
        for marker in markers:
            if marker in text:
                return pdf_idx
    log.warning(
        "no body-start marker found in first %d pages; "
        "falling back to page_skip_after_toc=%d",
        scan_limit, config.body.page_skip_after_toc,
    )
    return min(config.body.page_skip_after_toc, len(doc) - 1)


def _split_title_and_body(line: str) -> tuple[str, str]:
    """Split a section-title line into (title, body_remainder).

    Some doc families (e.g. credit agreements) pack the first sentence of
    the body onto the title line, separated by `.<2+spaces>`. For docs
    without that style, no split occurs and body_remainder is "".
    """
    m = re.search(r"\.\s{2,}", line)
    if m:
        title = line[: m.start()].strip()
        body = line[m.end():].strip()
        return title.rstrip("."), body
    return line.rstrip("."), ""


def _finalize_section(state: dict) -> Section:
    return Section(
        chapter=state["chapter"],
        section_id=state["section_id"],
        section_title=state["section_title"],
        text="\n".join(state["text_lines"]).strip(),
        page_start=state["page_start"],
        page_end=state["page_end"],
    )
