"""LangGraph agentic loop that produces a per-document parsing config.

The loop: propose → validate → smoke-test → critique → refine. Gemini Flash
3.5 proposes pattern overrides; Pydantic catches invalid regexes; the actual
parser is run as the reward signal; a hybrid rule+LLM critic decides whether
to accept or revise. No config reaches the vector DB until the parser
metrics actually pass.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Optional, TypedDict

import fitz
import yaml
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from RAG.chunker import count_tokens
from RAG.pdf_parser import Section, parse_pdf
from RAG.pre_injection.config import (
    ParsedConfig,
    _CONFIGS_DIR,
    _deep_merge,
    _load_chain,
    load_config,
)

# Load .env from cwd or any parent dir so GEMINI_API_KEY / GOOGLE_API_KEY
# is available when _llm() is called — works whether this module is run
# via the CLI (RAG.pre_injection.configure) or imported from a notebook.
load_dotenv()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Override schema — what the LLM fills in. Mirrors ParsedConfig but every
# field is Optional so the LLM only emits what differs from defaults.
# ---------------------------------------------------------------------------

class TocOverride(BaseModel):
    heading_pattern: Optional[str] = None
    entry_patterns: Optional[list[str]] = None
    chapter_pattern: Optional[str] = None
    search_pages: Optional[int] = None
    blank_page_streak_stop: Optional[int] = None
    min_sections_to_accept: Optional[int] = None


class BodyOverride(BaseModel):
    section_header_pattern: Optional[str] = None
    section_id_only_pattern: Optional[str] = None
    chapter_pattern: Optional[str] = None
    chapter_allcaps_pattern: Optional[str] = None
    schedule_cutoff_pattern: Optional[str] = None
    body_start_markers: Optional[list[str]] = None
    line_skip_patterns: Optional[list[str]] = None
    page_skip_after_toc: Optional[int] = None


class PageNumberingOverride(BaseModel):
    printed_page_pattern: Optional[str] = None
    search_first_n_lines: Optional[int] = None
    search_last_n_lines: Optional[int] = None


class HeaderFooterStripOverride(BaseModel):
    enable: Optional[bool] = None
    min_span_pages: Optional[int] = None
    min_line_length: Optional[int] = None
    max_line_length: Optional[int] = None
    min_repetition_fraction: Optional[float] = None


class DefinitionsOverride(BaseModel):
    section_id_hints: Optional[list[str]] = None
    title_prefix_hints: Optional[list[str]] = None
    term_split_pattern: Optional[str] = None
    term_head_pattern: Optional[str] = None


class ConfigOverride(BaseModel):
    """Slot-fill schema for Gemini. Every field optional; only emit what
    differs from RAG/pre_injection/configs/_defaults.yaml."""
    toc: Optional[TocOverride] = Field(
        default=None, description="TOC parsing patterns and thresholds."
    )
    body: Optional[BodyOverride] = Field(
        default=None, description="Body section header patterns and skip rules."
    )
    page_numbering: Optional[PageNumberingOverride] = Field(
        default=None, description="Printed page-number detection."
    )
    header_footer_strip: Optional[HeaderFooterStripOverride] = Field(
        default=None, description="Repeated header/footer line filtering."
    )
    definitions: Optional[DefinitionsOverride] = Field(
        default=None,
        description=(
            "Definitions section identification and term-split patterns. "
            "term_split_pattern uses a lookahead (?=...) anchored to ^ in "
            "multiline mode (?m) to split on each defined-term head."
        ),
    )


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class SmokeMetrics(TypedDict):
    section_count: int
    has_definitions: bool
    definitions: dict | None  # {section_id, title, page_start, page_end, char_count, token_count}
    avg_tokens: int
    min_tokens: int
    max_tokens: int
    largest_section_pages: int
    pages_covered: tuple[int, int]
    total_pages: int
    largest_section_pct: float
    first_titles: list[str]
    empty_sections: int
    term_chunks: int
    anchor_match_rate: float  # frac of sections whose id/title sits on its page_start


class Iteration(TypedDict):
    proposed_config: dict
    smoke_metrics: SmokeMetrics | None
    validation_error: str | None
    critique: str


class SectionSummary(TypedDict):
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    token_count: int


class ConfigureState(TypedDict, total=False):
    pdf_path: str
    doc_id: str
    max_iterations: int
    front_matter_pages: int

    front_matter_text: str
    probe_samples: dict[str, str]

    proposed_config: dict | None
    validation_error: str | None
    smoke_metrics: SmokeMetrics | None
    section_summaries: list[SectionSummary] | None
    critique: str | None

    history: list[Iteration]
    iteration: int
    accepted: bool
    final_config_path: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

_MODEL_NAME = "gemini-3.5-flash"


def _llm() -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY or GEMINI_API_KEY env var is required"
        )
    return ChatGoogleGenerativeAI(
        model=_MODEL_NAME, temperature=0.2, google_api_key=api_key
    )


def extract_front_matter(state: ConfigureState) -> dict[str, Any]:
    n_pages = state.get("front_matter_pages", 20)
    doc = fitz.open(state["pdf_path"])
    try:
        end = min(len(doc), n_pages)
        chunks = []
        for i in range(end):
            chunks.append(f"=== PDF PAGE {i + 1} ===\n{doc[i].get_text('text')}")
        text = "\n".join(chunks)
    finally:
        doc.close()
    log.info("extract_front_matter: %d pages, %d chars", end, len(text))
    return {
        "front_matter_text": text,
        "probe_samples": {},
        "history": [],
        "iteration": 0,
        "accepted": False,
    }


def _baseline_yaml_for_prompt() -> str:
    """Return the current _defaults.yaml content so the LLM can see what
    it's inheriting. Cached at module load — defaults don't change at
    runtime."""
    return yaml.safe_dump(
        _load_chain("_defaults"), sort_keys=False, default_flow_style=False
    )


_BASELINE_YAML = _baseline_yaml_for_prompt()


_SYSTEM_PROMPT = f"""\
You are configuring a deterministic PDF parser for a legal financial
document (CLO indenture, CLO offering circular, or syndicated credit
agreement). These documents share a structure: a Table of Contents, then
Articles/Chapters subdivided into numbered Sections (e.g. "1.1", "1.01",
"2.10"). One section — typically Section 1.1 or 1.01 — is "Defined Terms"
or "Definitions" and runs 50-100 pages, with each defined term introduced
by a quoted name followed by either a colon (CLO style) or a verb phrase
like "means" / "has the meaning" (credit-agreement style).

Your job: read the supplied first ~20 pages of one such PDF and emit
overrides to the baseline config. The baseline is shown below — it
already handles the most common CLO indenture layout. Most PDFs need
ZERO or ONE override.

BASELINE CONFIG (this is what runs if you emit no overrides):
```yaml
{_BASELINE_YAML}```

CRITICAL RULES:
1. For each field, FIRST check whether the baseline pattern above
   already matches the lines you see in this PDF. If it does, leave
   that field null. Do NOT re-emit the baseline pattern as an override.
2. A regex that is LESS strict than the baseline (e.g. `(.*)` instead
   of `[A-Z][A-Za-z0-9 ,;:&\\-]+?`) is almost always wrong — it will
   match cross-references in body prose and create false section
   headers. Prefer to keep the baseline's restrictive character classes.
3. Only override `body_start_markers` and `line_skip_patterns` if the
   PDF clearly needs them; defaults are empty lists for a reason.
4. If you cannot identify a specific reason to deviate from the baseline
   for some group, emit nothing for that group.

Each regex you write will be compiled with re.compile and matched
line-by-line against page text extracted by pymupdf.

REGEX CAPTURE GROUP CONTRACT — every pattern field below MUST have the
specified number of capture groups `(...)` in the listed order. The
parser calls `m.group(N)` on these, so a missing group is a hard
validation error (NOT a warning). Lookaheads `(?=...)` and non-capturing
groups `(?:...)` do NOT count as capture groups.

  toc.chapter_pattern              → 3 groups: (chapter_num)(title)(page_num)
  toc.entry_patterns[*]            → 3 groups: (section_id)(title)(page_num)
  body.section_header_pattern      → 2 groups: (section_id)(title)
  body.section_id_only_pattern     → 1 group:  (section_id)
  body.chapter_pattern             → 1 group:  (chapter_num)
  body.chapter_allcaps_pattern     → 2 groups: (chapter_num)(title)
  definitions.term_split_pattern   → 1 group:  (term_name)
  definitions.term_head_pattern    → 1 group:  (term_name)

Fields with NO capture-group requirement (compile-only): toc.heading_pattern,
body.schedule_cutoff_pattern, body.line_skip_patterns[*],
page_numbering.printed_page_pattern.

Example — term_split_pattern with a lookahead AND a capture group:
  (?m)^(?=["“]([^"”]+)["”]\\s*:)    # captures term_name inside the lookahead

Guidance:
- Inline flags like (?i) and (?m) are supported and often needed.
- Test your patterns mentally against actual lines from the PDF text.
- For body_start_markers, supply short literal strings (NOT regex) that
  uniquely identify the first body page (e.g. "PRELIMINARY STATEMENTS",
  "This CREDIT AGREEMENT"). These are checked via substring match on
  each line.
- For line_skip_patterns, list any per-page running-footer or watermark
  regex you observe (e.g. "^\\s*US-DOCS\\\\" for the "US-DOCS\\..." stamp).
- Quoted defined-term names may use straight " or curly “”/'' glyphs;
  the baseline character class is ["“][^"”]+["”] — adjust if this PDF
  uses something different.
- If the section header style is "Section X.YY Title." with a trailing
  period (typical credit agreement), override section_header_pattern
  AND section_id_only_pattern so cross-references like "Section 6.02(a)."
  don't get treated as headers.
"""


def _format_history_for_llm(history: list[Iteration]) -> str:
    if not history:
        return ""
    parts = []
    for i, it in enumerate(history, start=1):
        m = it.get("smoke_metrics")
        m_str = "n/a" if m is None else (
            f"sections={m['section_count']} defs={m['has_definitions']} "
            f"avg_tokens={m['avg_tokens']} largest_pct={m['largest_section_pct']:.0%} "
            f"anchor_match={m['anchor_match_rate']:.0%}"
        )
        ve = it.get("validation_error")
        parts.append(
            f"--- attempt {i} ---\n"
            f"override: {json.dumps(it['proposed_config'], indent=2)}\n"
            f"metrics: {m_str}\n"
            f"{'validation_error: ' + ve if ve else ''}\n"
            f"critique: {it['critique']}"
        )
    return "PRIOR ATTEMPTS (avoid repeating these mistakes):\n" + "\n\n".join(parts)


def _format_probes(probes: dict[str, str]) -> str:
    if not probes:
        return ""
    parts = [f"--- {label} ---\n{text}" for label, text in probes.items()]
    return "ADDITIONAL PAGE SAMPLES FROM LATER IN THE DOC:\n\n" + "\n\n".join(parts)


def propose_config(state: ConfigureState) -> dict[str, Any]:
    iteration = state.get("iteration", 0) + 1
    llm = _llm().with_structured_output(ConfigOverride)

    user_parts = [
        f"PDF first {state.get('front_matter_pages', 20)} pages:\n\n"
        f"{state['front_matter_text']}",
    ]
    probes = state.get("probe_samples", {})
    if probes:
        user_parts.append(_format_probes(probes))
    history = state.get("history", [])
    if history:
        user_parts.append(_format_history_for_llm(history))

    messages = [
        ("system", _SYSTEM_PROMPT),
        ("user", "\n\n".join(user_parts)),
    ]
    log.info("propose_config: iteration %d (history=%d)", iteration, len(history))
    override: ConfigOverride = llm.invoke(messages)
    override_dict = override.model_dump(exclude_none=True)
    log.info(
        "propose_config: emitted overrides for groups: %s",
        sorted(override_dict.keys()),
    )
    return {
        "proposed_config": override_dict,
        "iteration": iteration,
        "validation_error": None,
    }


def validate_node(state: ConfigureState) -> dict[str, Any]:
    """Materialize the override into a full ParsedConfig (which triggers
    regex compilation + Pydantic validation). Failure appends a stub
    iteration to history so the next propose call sees the exact error
    message and can fix it instead of guessing again."""
    try:
        _materialize(state["proposed_config"])
        return {"validation_error": None}
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        log.warning("validate: %s", e)
        history = list(state.get("history", []))
        history.append({
            "proposed_config": state["proposed_config"],
            "smoke_metrics": None,
            "validation_error": msg,
            "critique": "(skipped — validation failed)",
        })
        return {"validation_error": msg, "history": history}


def _materialize(override_dict: dict) -> ParsedConfig:
    defaults = _load_chain("_defaults")
    merged = _deep_merge(defaults, override_dict)
    return ParsedConfig.model_validate(merged)


def smoke_test_node(state: ConfigureState) -> dict[str, Any]:
    """Run the actual parser with the proposed config and collect metrics.
    Parser crashes are caught and re-surfaced as validation_error so the
    next propose iteration can see the message and fix the patterns."""
    config = _materialize(state["proposed_config"])
    try:
        sections = parse_pdf(state["pdf_path"], config)
        doc = fitz.open(state["pdf_path"])
        total_pages = len(doc)
        try:
            metrics = _compute_smoke_metrics(sections, total_pages, doc)
            probes = _collect_probe_samples(doc, sections)
        finally:
            doc.close()
    except Exception as e:
        msg = f"parser crashed: {type(e).__name__}: {e}"
        log.warning("smoke_test: %s", msg)
        history = list(state.get("history", []))
        history.append({
            "proposed_config": state["proposed_config"],
            "smoke_metrics": None,
            "validation_error": msg,
            "critique": "(skipped — parser crashed during smoke test)",
        })
        return {"validation_error": msg, "smoke_metrics": None, "history": history}
    section_summaries: list[SectionSummary] = [
        {
            "section_id": s["section_id"],
            "section_title": s["section_title"],
            "page_start": s["page_start"],
            "page_end": s["page_end"],
            "token_count": count_tokens(s["text"]),
        }
        for s in sections
    ]
    log.info(
        "smoke_test: sections=%d defs=%s avg_tokens=%d largest_pct=%.0f%% anchor=%.0f%%",
        metrics["section_count"], metrics["has_definitions"],
        metrics["avg_tokens"], metrics["largest_section_pct"] * 100,
        metrics["anchor_match_rate"] * 100,
    )
    return {
        "smoke_metrics": metrics,
        "probe_samples": probes,
        "section_summaries": section_summaries,
    }


def _section_anchor_match_rate(
    doc: fitz.Document, sections: list[Section]
) -> float:
    """Fraction of (non-empty) sections whose own section_id or title text
    actually appears on the PDF page the parser assigned as page_start
    (checked over page_start ± 1 to tolerate boundary pages).

    This is the decisive guard against a bad page offset: when section
    page ranges are shifted onto the wrong part of the document (the
    cpi.pdf +307 bug), almost no section's header lands on its claimed
    start page, so this rate collapses toward 0. A correct parse scores
    high (most section titles appear verbatim as their body header)."""
    non_empty = [s for s in sections if s["text"].strip()]
    if not non_empty:
        return 0.0
    matched = 0
    for s in non_empty:
        ps = s["page_start"] - 1  # to 0-based
        texts = [
            doc[i].get_text("text")
            for i in (ps - 1, ps, ps + 1)
            if 0 <= i < len(doc)
        ]
        page_text = "\n".join(texts)
        sid = (s["section_id"] or "").strip()
        title = (s["section_title"] or "").strip()
        if sid and re.search(r"(?:Section\s+)?" + re.escape(sid) + r"\b", page_text):
            matched += 1
        elif len(title) >= 6 and title[:40] in page_text:
            matched += 1
    return matched / len(non_empty)


def _compute_smoke_metrics(
    sections: list[Section], total_pages: int, doc: fitz.Document
) -> SmokeMetrics:
    non_empty = [s for s in sections if s["text"].strip()]
    if not non_empty:
        return {
            "section_count": len(sections),
            "has_definitions": False,
            "definitions": None,
            "avg_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "largest_section_pages": 0,
            "pages_covered": (0, 0),
            "total_pages": total_pages,
            "largest_section_pct": 0.0,
            "first_titles": [],
            "empty_sections": len(sections),
            "term_chunks": 0,
            "anchor_match_rate": 0.0,
        }
    tokens = [count_tokens(s["text"]) for s in non_empty]
    pages = [s["page_end"] - s["page_start"] + 1 for s in non_empty]
    page_min = min(s["page_start"] for s in non_empty)
    page_max = max(s["page_end"] for s in non_empty)
    largest = max(pages)

    # Heuristic: definitions = section with most tokens (since it spans 50-100
    # pages of dense text in this corpus). Cross-check with title.
    defs_section = max(non_empty, key=lambda s: count_tokens(s["text"]))
    has_defs_by_size = count_tokens(defs_section["text"]) > 5000
    has_defs_by_title = defs_section["section_title"].lower().startswith("defin")

    return {
        "section_count": len(sections),
        "has_definitions": has_defs_by_size and has_defs_by_title,
        "definitions": {
            "section_id": defs_section["section_id"],
            "title": defs_section["section_title"],
            "page_start": defs_section["page_start"],
            "page_end": defs_section["page_end"],
            "char_count": len(defs_section["text"]),
            "token_count": count_tokens(defs_section["text"]),
        },
        "avg_tokens": int(statistics.mean(tokens)),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
        "largest_section_pages": largest,
        "pages_covered": (page_min, page_max),
        "total_pages": total_pages,
        "largest_section_pct": largest / total_pages if total_pages else 0.0,
        "first_titles": [s["section_title"] for s in non_empty[:10]],
        "empty_sections": len(sections) - len(non_empty),
        "term_chunks": 0,  # filled in by chunker stage if/when we add it
        "anchor_match_rate": _section_anchor_match_rate(doc, sections),
    }


def _collect_probe_samples(
    doc: fitz.Document, sections: list[Section]
) -> dict[str, str]:
    """Pull 1-2 text samples from later in the doc to verify patterns hold
    beyond the front matter."""
    samples: dict[str, str] = {}
    if not sections:
        return samples
    # Mid-definitions page (if we have a definitions-like section)
    defs = max(sections, key=lambda s: len(s["text"]), default=None)
    if defs and defs["page_end"] > defs["page_start"]:
        mid = (defs["page_start"] + defs["page_end"]) // 2 - 1
        if 0 <= mid < len(doc):
            txt = doc[mid].get_text("text")
            samples[f"mid of section {defs['section_id']!r} (PDF page {mid + 1})"] = (
                txt[:2500]
            )
    # A random body page from the second half
    if len(doc) > 50:
        rnd = random.Random(0).randrange(len(doc) // 2, len(doc) - 2)
        samples[f"random body page {rnd + 1}"] = doc[rnd].get_text("text")[:2500]
    return samples


# ---------------------------------------------------------------------------
# Critique — hybrid rule + LLM
# ---------------------------------------------------------------------------

_RULE_THRESHOLDS = {
    "min_sections": 10,
    "max_largest_pct": 0.50,
    "min_avg_tokens": 200,
    "max_avg_tokens": 50_000,
    "min_anchor_match": 0.50,
}


def _rule_critique(m: SmokeMetrics) -> str | None:
    """Return a critique string if any rule fails; None if all rules pass."""
    if m["section_count"] < _RULE_THRESHOLDS["min_sections"]:
        return (
            f"Only {m['section_count']} sections parsed (need at least "
            f"{_RULE_THRESHOLDS['min_sections']}). Likely the section header "
            f"pattern is wrong or body_start_markers is misplaced."
        )
    if m["anchor_match_rate"] < _RULE_THRESHOLDS["min_anchor_match"]:
        return (
            f"Only {m['anchor_match_rate']:.0%} of sections have their own "
            f"id/title on their assigned start page (need at least "
            f"{_RULE_THRESHOLDS['min_anchor_match']:.0%}). The page-number "
            f"offset is almost certainly wrong — section page ranges are "
            f"mapped onto the wrong part of the document (common when the PDF "
            f"concatenates several documents that each restart page numbering, "
            f"or when printed_page_pattern / search_first_n_lines miss the "
            f"page number). Verify page_numbering settings."
        )
    if not m["has_definitions"]:
        return (
            "No definitions section detected by size + title heuristic. "
            "Verify section_id_hints/title_prefix_hints and confirm the "
            "definitions section is being extracted (it should be ~50-100 "
            "pages and start with 'Defin...')."
        )
    if m["largest_section_pct"] > _RULE_THRESHOLDS["max_largest_pct"]:
        return (
            f"Largest section spans {m['largest_section_pct']:.0%} of the doc — "
            f"likely a section boundary regex is missing matches and one "
            f"section is swallowing many others. Tighten section_header_pattern "
            f"or fix schedule_cutoff_pattern."
        )
    if m["avg_tokens"] < _RULE_THRESHOLDS["min_avg_tokens"]:
        return (
            f"Average section size is only {m['avg_tokens']} tokens — "
            f"sections are getting fragmented. The section_header_pattern "
            f"is probably matching non-header lines."
        )
    if m["avg_tokens"] > _RULE_THRESHOLDS["max_avg_tokens"]:
        return (
            f"Average section size {m['avg_tokens']} tokens is too large — "
            f"section boundaries are being missed."
        )
    return None


def critique_node(state: ConfigureState) -> dict[str, Any]:
    m = state["smoke_metrics"]
    assert m is not None, "smoke_test must run before critique"

    rule_msg = _rule_critique(m)
    if rule_msg is not None:
        critique = f"REVISE: {rule_msg}"
    else:
        # Rule layer passed; defer to LLM for semantic check.
        critique = _llm_critique(state)

    log.info("critique: %s", critique[:200])
    # Append this iteration to history regardless of accept/revise
    history = list(state.get("history", []))
    history.append({
        "proposed_config": state["proposed_config"],
        "smoke_metrics": m,
        "validation_error": state.get("validation_error"),
        "critique": critique,
    })
    return {"critique": critique, "history": history}


_CRITIC_SYSTEM = """\
You are a critic for a deterministic PDF parser config. You will see
the metrics of a parse and a sample of the parsed section titles. Decide
whether the section structure looks semantically correct or whether the
config needs revising.

Reply with EXACTLY one of:
  ACCEPT
  REVISE: <one-sentence specific instruction>

Look for: nonsense titles (single words, capitalization issues, dot-leader
fragments), missing major sections, definitions section that's clearly
truncated or split across sections.
"""


def _llm_critique(state: ConfigureState) -> str:
    m = state["smoke_metrics"]
    assert m is not None
    llm = _llm()
    user = (
        f"Section count: {m['section_count']}\n"
        f"Average tokens per section: {m['avg_tokens']}\n"
        f"Definitions section: {json.dumps(m['definitions'], indent=2)}\n"
        f"First 10 section titles: {json.dumps(m['first_titles'], indent=2)}\n"
    )
    resp = llm.invoke([("system", _CRITIC_SYSTEM), ("user", user)])
    content = resp.content
    if isinstance(content, list):
        # Gemini sometimes returns content as a list of blocks.
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(getattr(block, "text", str(block)))
        text = "".join(parts).strip()
    else:
        text = content.strip()
    if text.upper().startswith("ACCEPT"):
        return "ACCEPT"
    if text.upper().startswith("REVISE"):
        return text
    return f"REVISE: {text}"  # unknown form — treat as revise


# ---------------------------------------------------------------------------
# Human in the loop — runs after the LLM critic emits ACCEPT
# ---------------------------------------------------------------------------

def _print_review_table(state: ConfigureState) -> None:
    m = state["smoke_metrics"]
    summaries = state.get("section_summaries") or []
    print("\n" + "=" * 78)
    print(f"HUMAN REVIEW — iteration {state['iteration']} of {state.get('max_iterations', 5)}")
    print("=" * 78)
    print("\nSmoke metrics:")
    print(f"  Sections parsed:        {m['section_count']}")
    print(f"  Has definitions:        {m['has_definitions']}")
    if m.get("definitions"):
        d = m["definitions"]
        print(
            f"  Definitions section:    {d['section_id']!r} '{d['title']}' "
            f"pp {d['page_start']}–{d['page_end']} ({d['token_count']} tokens)"
        )
    print(
        f"  Pages covered:          {m['pages_covered'][0]}–{m['pages_covered'][1]} "
        f"of {m['total_pages']}"
    )
    print(
        f"  Tokens avg/min/max:     {m['avg_tokens']} / {m['min_tokens']} / {m['max_tokens']}"
    )
    print(
        f"  Largest section:        {m['largest_section_pages']} pages "
        f"({m['largest_section_pct']:.0%})"
    )
    print(f"  Empty sections:         {m['empty_sections']}")
    print(f"  Anchor match rate:      {m['anchor_match_rate']:.0%}")
    print(f"\nAll {len(summaries)} parsed sections:")
    print(f"  {'section_id':>12s}  {'pages':>10s}  {'tokens':>7s}  title")
    print(f"  {'-'*12}  {'-'*10}  {'-'*7}  {'-'*55}")
    for s in summaries:
        page_str = f"{s['page_start']}-{s['page_end']}"
        title = (s["section_title"] or "")[:55]
        print(
            f"  {s['section_id'] or '':>12s}  {page_str:>10s}  "
            f"{s['token_count']:>7d}  {title}"
        )
    print("\nProposed override:")
    print(yaml.safe_dump(state["proposed_config"], sort_keys=False, default_flow_style=False))


def human_review_node(state: ConfigureState) -> dict[str, Any]:
    """Show the parse stats and proposed config; let the user accept or
    revise. On revise, the user's reason overwrites the last history
    entry's critique and routes back to propose (subject to max-iter)."""
    _print_review_table(state)
    while True:
        choice = input("Action — [a]ccept / [r]evise? ").strip().lower()
        if choice in ("a", "accept", ""):
            log.info("human_review: user accepted")
            return {"critique": "ACCEPT (confirmed by user)"}
        if choice in ("r", "revise"):
            reason = input("Reason for revision (becomes the next critique): ").strip()
            if not reason:
                reason = "user requested revision (no reason given)"
            critique = f"REVISE: {reason}"
            log.info("human_review: user revised — %s", reason)
            # Overwrite the last history entry's critique so propose sees
            # the human's verdict, not the LLM critic's stale ACCEPT.
            history = list(state.get("history", []))
            if history:
                last = dict(history[-1])
                last["critique"] = critique
                history[-1] = last
            return {"critique": critique, "history": history}
        print("Please enter 'a' or 'r'.")


# ---------------------------------------------------------------------------
# Terminal nodes
# ---------------------------------------------------------------------------

def accept_node(state: ConfigureState) -> dict[str, Any]:
    out_path = _CONFIGS_DIR / f"{state['doc_id']}.yaml"
    payload = {"extends": "_defaults", **state["proposed_config"]}
    header = (
        f"# Auto-generated by RAG.pre_injection.configure on PDF {state['pdf_path']}\n"
        f"# Iterations: {state['iteration']}\n"
        f"# Final smoke metrics: "
        f"sections={state['smoke_metrics']['section_count']}, "
        f"defs={state['smoke_metrics']['has_definitions']}, "
        f"avg_tokens={state['smoke_metrics']['avg_tokens']}\n"
    )
    with out_path.open("w") as f:
        f.write(header)
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    log.info("accept: wrote %s", out_path)
    # Verify the written file loads cleanly
    load_config(state["doc_id"])
    return {"accepted": True, "final_config_path": str(out_path)}


def give_up_node(state: ConfigureState) -> dict[str, Any]:
    out_path = _CONFIGS_DIR / f"{state['doc_id']}.draft.yaml"
    payload = {"extends": "_defaults", **(state.get("proposed_config") or {})}
    header = (
        f"# DRAFT — RAG.pre_injection.configure did not converge after {state['iteration']} "
        f"iterations on {state['pdf_path']}\n"
        f"# This file is NOT picked up by load_config (the `.draft.` infix "
        f"keeps it inert). Inspect history, fix by hand, then rename.\n"
    )
    with out_path.open("w") as f:
        f.write(header)
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)

        # Embed history for debugging
        f.write("\n# === iteration history ===\n")
        for i, it in enumerate(state.get("history", []), start=1):
            f.write(f"#\n# --- iter {i} ---\n")
            f.write(f"# critique: {it['critique']}\n")
            if it.get("validation_error"):
                f.write(f"# validation_error: {it['validation_error']}\n")
            m = it.get("smoke_metrics")
            if m is not None:
                f.write(
                    f"# metrics: sections={m['section_count']} "
                    f"defs={m['has_definitions']} avg_tokens={m['avg_tokens']} "
                    f"largest_pct={m['largest_section_pct']:.0%}\n"
                )
    log.warning("give_up: wrote draft to %s", out_path)
    return {"accepted": False, "final_config_path": str(out_path)}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_validate(state: ConfigureState) -> str:
    if not state.get("validation_error"):
        return "smoke_test"
    if state.get("iteration", 0) >= state.get("max_iterations", 5):
        return "give_up"
    return "propose"


def _route_after_smoke_test(state: ConfigureState) -> str:
    if not state.get("validation_error"):
        return "critique"
    if state.get("iteration", 0) >= state.get("max_iterations", 5):
        return "give_up"
    return "propose"


def _route_after_critique(state: ConfigureState) -> str:
    critique = state.get("critique") or ""
    if critique.upper().startswith("ACCEPT"):
        return "human_review"
    # Stall detection: if the last two proposed configs are identical, give up.
    history = state.get("history", [])
    if len(history) >= 2:
        a = json.dumps(history[-1]["proposed_config"], sort_keys=True)
        b = json.dumps(history[-2]["proposed_config"], sort_keys=True)
        if a == b:
            log.warning("route: identical configs back-to-back, giving up")
            return "give_up"
    if state.get("iteration", 0) >= state.get("max_iterations", 5):
        return "give_up"
    return "propose"


def _route_after_human_review(state: ConfigureState) -> str:
    critique = state.get("critique") or ""
    if critique.upper().startswith("ACCEPT"):
        return "accept"
    if state.get("iteration", 0) >= state.get("max_iterations", 5):
        return "give_up"
    return "propose"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(ConfigureState)
    g.add_node("extract", extract_front_matter)
    g.add_node("propose", propose_config)
    g.add_node("validate", validate_node)
    g.add_node("smoke_test", smoke_test_node)
    g.add_node("critique", critique_node)
    g.add_node("human_review", human_review_node)
    g.add_node("accept", accept_node)
    g.add_node("give_up", give_up_node)

    g.set_entry_point("extract")
    g.add_edge("extract", "propose")
    g.add_edge("propose", "validate")
    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"propose": "propose", "smoke_test": "smoke_test", "give_up": "give_up"},
    )
    g.add_conditional_edges(
        "smoke_test",
        _route_after_smoke_test,
        {"critique": "critique", "propose": "propose", "give_up": "give_up"},
    )
    g.add_conditional_edges(
        "critique",
        _route_after_critique,
        {
            "propose": "propose",
            "human_review": "human_review",
            "give_up": "give_up",
        },
    )
    g.add_conditional_edges(
        "human_review",
        _route_after_human_review,
        {"propose": "propose", "accept": "accept", "give_up": "give_up"},
    )
    g.add_edge("accept", END)
    g.add_edge("give_up", END)
    return g.compile()
