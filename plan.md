# Refactor `RAG/` to use externalized per-document configs

## Context

`pdf_parser.py` and `chunker.py` are currently coupled to two specific document types (CLO indenture, credit agreement) via a `doc_type` string and hardcoded regex constants ([pdf_parser.py:30-67](RAG/pdf_parser.py#L30-L67), [pdf_parser.py:93-101](RAG/pdf_parser.py#L93-L101), [chunker.py:36-67](RAG/chunker.py#L36-L67), [chunker.py:138-141](RAG/chunker.py#L138-L141)). Every new document family — offering circulars, other credit agreements, etc. — requires a new branch and new constants. The corpus we care about (CLO indentures, CLO offering circulars, credit agreements) shares structure (TOC + Article/Section hierarchy + ~90-page definitions block) but differs on surface patterns (quote glyphs, definition delimiter, body-start markers, exact section header form).

**Goal**: make `pdf_parser.py` and `chunker.py` document-agnostic — pure deterministic engines that read regexes and thresholds from an external YAML config. For each new PDF, an LLM-driven `configure.py` reads the first 20 pages and produces an override config; the parser/chunker then run unchanged. The *algorithm* (TOC-first parsing, page-offset detection, header/footer stripping, definitions-aware chunking) stays in code; only *patterns and thresholds* move to config.

## Design overview

**One config schema covers both parser and chunker** — they share document-format assumptions, so externalizing the parser alone would leave doc-type leakage in the chunker.

**Two-layer config**: `configs/_defaults.yaml` holds patterns that work for most CLO/credit legal documents. Each PDF gets a `configs/<doc_id>.yaml` that uses `extends: _defaults` and overrides only what differs. This matches the reality of law-firm boilerplate reuse and keeps per-PDF configs small.

**Unify the parsing strategy**: today, `clo_indenture` uses TOC-first and `credit_agreement` uses linewalk-with-body-start-markers. Since you confirmed all target documents have a TOC, the new parser is always TOC-first with linewalk fallback (existing behavior at [pdf_parser.py:111-112](RAG/pdf_parser.py#L111-L112)). Body-start markers (`"This CREDIT AGREEMENT"`, etc.) move into config as `body.body_start_markers` and are used by the linewalk fallback to skip front matter — no more strategy branching.

**LLM-config workflow**: `configure.py` extracts the first 20 pages, calls the Anthropic API with the Pydantic config schema as a tool definition (LLM fills slots, doesn't free-write YAML), validates against the schema, writes `configs/<doc_id>.yaml`, then **smoke-tests by actually running `parse_pdf`** and prints diagnostics (section count, definitions section detected with page range, avg/min/max section size in tokens, any pages absorbed by a single oversized section). If diagnostics look bad, you re-prompt with the feedback.

## Config schema (Pydantic, serialized as YAML)

Top-level groups, each independently overridable:

- `toc` — TOC heading regex, entry regexes (list, tried in order), chapter regex, `search_pages`, `blank_page_streak_stop`, `min_sections_to_accept`
- `body` — section_header / section_id_only / chapter / chapter_allcaps / schedule_cutoff regexes; `body_start_markers: list[str]` (anchors for linewalk fallback); `page_skip_after_toc`
- `page_numbering` — printed_page regex, `search_first_n_lines`, `search_last_n_lines`
- `header_footer_strip` — `enable`, `min_span_pages`, `min_line_length`, `max_line_length`, `min_repetition_fraction`
- `definitions` — `section_id_hints: list[str]`, `title_prefix_hints: list[str]`, `term_split_pattern`, `term_head_pattern` (this is the credit-agreement-vs-indenture difference in [chunker.py:36-67](RAG/chunker.py#L36-L67))

Inheritance: load `_defaults.yaml` → load override → deep-merge → validate via Pydantic → compile regexes once. Pydantic validators reject malformed regex at load time (so bad LLM output fails fast, not at parse time).

## File-by-file changes

### New: [RAG/config.py](RAG/config.py)
- Pydantic models for the schema above
- `load_config(doc_id_or_path: str | Path) -> ParsedConfig` — resolves `configs/<doc_id>.yaml` if it exists else `configs/_defaults.yaml`; handles `extends:` chain; deep-merges; validates; pre-compiles every regex into a `re.Pattern` attribute alongside the source string
- Validators: every regex field is compiled at load time; failure surfaces the field name

### New: [RAG/configs/_defaults.yaml](RAG/configs/_defaults.yaml)
- Baseline using current CLO-indenture patterns from [pdf_parser.py:30-67](RAG/pdf_parser.py#L30-L67) and CLO definitions patterns from [chunker.py:43-44](RAG/chunker.py#L43-L44)
- Numeric defaults: `search_pages=20`, `blank_page_streak_stop=2`, `min_sections_to_accept=5`, `min_span_pages=4`, `min_line_length=3`, `max_line_length=120`, `min_repetition_fraction=0.5`, `page_skip_after_toc=10`

### New: [RAG/configs/sample_credit_agreement.yaml](RAG/configs/sample_credit_agreement.yaml)
Worked example demonstrating override semantics:
```yaml
extends: _defaults
body:
  body_start_markers: ["This CREDIT AGREEMENT", "PRELIMINARY STATEMENTS"]
definitions:
  term_split_pattern: '(?m)(?=^\s*("[^"\n]+")\s+(?:means|has\s+the\s+meaning|shall\s+\w+|refers?\s+to)\b)'
  term_head_pattern: '^\s*("[^"\n]+")\s+(?:means|has\s+the\s+meaning|shall\s+\w+|refers?\s+to)\b'
```

### New: [RAG/configure_graph.py](RAG/configure_graph.py) — LangGraph agentic loop

Single-shot LLM-to-YAML is brittle because (a) LLMs produce near-miss regexes that compile but mis-match, and (b) the only reliable signal of correctness is *actually running the parser*. So `configure` is an **agentic loop**: propose → validate → smoke-test → critique → refine, with the parser's diagnostics as the reward signal. The config never reaches the vector DB until the loop's accept criteria pass — this directly addresses the user's concern about smoke-testing before injection.

**State** (TypedDict, threaded through the graph):
- `pdf_path: str`, `doc_id: str`
- `front_matter_text: str` — first 20 pages, extracted once
- `probe_samples: dict[str, str]` — text snippets pulled on demand from later pages (e.g. expected definitions page, a body-section page) to verify patterns hold past page 20
- `proposed_config: dict | None` — latest override (pre-Pydantic dict)
- `validation_error: str | None` — Pydantic / regex-compile error from last `validate` step
- `smoke_metrics: SmokeMetrics | None` — section count, definitions section (id + page range), avg/min/max section size in tokens, largest-section fraction of doc, pages covered, list of first 10 section titles
- `critique: str | None` — human-readable feedback on what to fix next iteration
- `history: list[Iteration]` — (config, metrics, critique) per round, for the LLM to see what it's already tried
- `iteration: int`, `accepted: bool`

**Nodes**:

1. **`extract_front_matter`** (runs once) — uses pymupdf to pull text from first 20 pages into `front_matter_text`.
2. **`propose_config`** — Google Gemini Flash 3.5 call (`gemini-flash-3.5`) via `langchain-google-genai`'s `ChatGoogleGenerativeAI`. System prompt explains the document family (CLO indenture / offering circular / credit agreement; ~90-page definitions block; TOC always present). User content: front-matter text + any probe samples + history of prior attempts and their critiques. The Pydantic config schema is bound to the model via `.with_structured_output(ConfigOverride)` so Gemini fills slots rather than free-writing YAML. Output: `proposed_config` dict. Reads `GOOGLE_API_KEY` from env.
3. **`validate`** — runs the dict through Pydantic models from [RAG/config.py](RAG/config.py); each regex field is `re.compile`d. On failure, populates `validation_error` and routes back to `propose_config` (LLM sees the error in next turn).
4. **`smoke_test`** — calls `parse_pdf(pdf_path, config)` and computes `SmokeMetrics`. Also pulls 1-2 probe samples (definitions section middle page, a random body page) into `probe_samples` for the next iteration.
5. **`critique`** — **hybrid rule-based + LLM**:
   - Rule layer (cheap, deterministic, runs first):
     - section_count ≥ `min_acceptable_sections` (default 10)
     - definitions section was identified (by `section_id_hints` or `title_prefix_hints` hit)
     - no single section absorbs >50% of total pages
     - no section is <100 chars (suggests a header-only false positive)
     - avg section size between 200 and 50k tokens
   - LLM layer (only if rules pass): given the first 10 section titles + a sample of the definitions section, judge whether section boundaries and term splits look semantically correct. Returns `accept` or `revise: <reason>`.
6. **`accept`** (terminal) — writes `configs/<doc_id>.yaml` with `extends: _defaults`, plus a header comment recording iteration count and final smoke metrics for audit. Sets `accepted=True`.
7. **`give_up`** (terminal) — reached on `iteration >= max_iterations` (default 5). Writes the best-scoring attempt to `configs/<doc_id>.draft.yaml` (note the `.draft.` infix — never picked up by `load_config`) and prints the full history so the user can hand-fix.

**Edges**:
- `START → extract_front_matter → propose_config`
- `propose_config → validate`
- `validate`: ok → `smoke_test`; error → `propose_config` (with error in state)
- `smoke_test → critique`
- `critique`: accept → `accept → END`; revise and `iteration < max` → `propose_config`; revise and `iteration >= max` → `give_up → END`

**Stall detection**: if two consecutive iterations produce byte-identical `proposed_config`, force a stronger prompt nudge (include `"You proposed the same config twice; vary the patterns"`); if it happens a third time, route to `give_up` early.

### New: [RAG/configure.py](RAG/configure.py) — thin CLI wrapper

`python -m RAG.configure <pdf_path> [--doc-id ID] [--pages 20] [--max-iter 5]`. Steps:
1. Compute `doc_id` from SHA1 if not provided (reuse logic from [rag.py:390](RAG/rag.py#L390))
2. Build the graph from `configure_graph.py`
3. Invoke with initial state; stream node events to stdout so each iteration is visible
4. Exit 0 if `accepted`, 1 if `give_up` (with path to the `.draft.yaml`)

### Modify: [RAG/pdf_parser.py](RAG/pdf_parser.py)
- Remove `doc_type` parameter; remove `_parse_clo_indenture` and `_parse_credit_agreement` ([pdf_parser.py:93-101](RAG/pdf_parser.py#L93-L101), [pdf_parser.py:106](RAG/pdf_parser.py#L106), [pdf_parser.py:465](RAG/pdf_parser.py#L465))
- Remove all module-level regex constants ([pdf_parser.py:30-67](RAG/pdf_parser.py#L30-L67)); they come from `config` now
- New signature: `parse_pdf(pdf_path: str, config: ParsedConfig) -> list[Section]`
- Unified algorithm: always run `_parse_via_toc`; if it returns fewer than `config.toc.min_sections_to_accept`, fall back to `_parse_via_linewalk`. The linewalk fallback consults `config.body.body_start_markers` (replaces `_ca_find_body_start` at [pdf_parser.py:580](RAG/pdf_parser.py#L580)) to skip front matter
- Helpers (`_detect_page_offset`, `_build_sections_from_toc`, `_trim_to_section_boundaries`, `_strip_running_headers_footers`, `_body_header_re`, `_find_schedule_cutoff`) keep their algorithms but read patterns/thresholds from `config` instead of module constants

### Modify: [RAG/chunker.py](RAG/chunker.py)
- Remove `doc_type` parameter ([chunker.py:74](RAG/chunker.py#L74), [chunker.py:138-141](RAG/chunker.py#L138-L141))
- Remove `_TERM_SPLIT_RE`, `_TERM_HEAD_RE`, `_CA_TERM_SPLIT_RE`, `_CA_TERM_HEAD_RE` ([chunker.py:36-67](RAG/chunker.py#L36-L67))
- New signature: `chunk_sections(sections, doc_id, config: ParsedConfig, max_tokens=2048, chunk_overlap_tokens=128, definitions_intro_tokens=500) -> list[Chunk]`
- `_is_definitions_section` ([chunker.py:125](RAG/chunker.py#L125)) reads `config.definitions.section_id_hints` and `title_prefix_hints` instead of hardcoded `"defin"` / `"1.1"`
- `_split_definitions` ([chunker.py:134](RAG/chunker.py#L134)) uses `config.definitions.term_split_pattern` and `term_head_pattern`

### Modify: [RAG/rag.py](RAG/rag.py)
- Drop `doc_type` from `ingest` ([rag.py:94](RAG/rag.py#L94)); add `config_path: str | Path | None = None`
- Drop `toc_search_pages` and `toc_pages` from the constructor ([rag.py:35-36](RAG/rag.py#L35-L36)) — they move into the config
- In `ingest`: resolve config via `load_config(config_path or doc_id)`; if no per-doc config exists, fall back to `_defaults` with a warning log
- Pass the resolved config object to `parse_pdf` and `chunk_sections`
- Other RAG knobs (`max_chunk_tokens`, `chunk_overlap_tokens`, model names, device) stay as constructor args — they're runtime/infrastructure, not document-format

### Modify: [RAG/testing/_test_step1_parse.py](RAG/testing/_test_step1_parse.py), [RAG/testing/_test_step2_ingest.py](RAG/testing/_test_step2_ingest.py)
- Update calls to use the new config-driven API
- `_test_step2_ingest.py` ([RAG/testing/_test_step2_ingest.py:53-58](RAG/testing/_test_step2_ingest.py#L53-L58)) currently passes `doc_type` — replace with `config_path` pointing at the appropriate config file
- `_inspect_chunks.py` ([RAG/testing/_inspect_chunks.py:26](RAG/testing/_inspect_chunks.py#L26)) — pass a config object instead of relying on the implicit indenture default

### Modify: [RAG/requirements.txt](RAG/requirements.txt)
Add `pyyaml>=6.0`, `pydantic>=2.6`, `langgraph>=0.2`, `langchain-google-genai>=2.0`, `google-generativeai>=0.8`. The latter three are only used by `configure.py` / `configure_graph.py`, not by the parse/chunk/ingest path.

## Reuse / don't reinvent

- SHA1 doc_id logic ([rag.py:101-102, 390](RAG/rag.py#L101-L102)) — reuse in `configure.py`
- `_strip_running_headers_footers` ([pdf_parser.py:314](RAG/pdf_parser.py#L314)) — algorithm stays, only thresholds become config-driven
- `_detect_page_offset` ([pdf_parser.py:203](RAG/pdf_parser.py#L203)) — algorithm stays; only the printed-page regex and search-line counts become config-driven
- `RecursiveCharacterTextSplitter` setup in `_recursive_subsplit` ([chunker.py:181](RAG/chunker.py#L181)) — no change needed (separators `["\n\n"]` are not document-specific enough to warrant exposing yet)

## Verification

1. **Schema sanity**: `python -c "from RAG.config import load_config; print(load_config('_defaults'))"` — should print a fully-populated `ParsedConfig` with every regex pre-compiled.
2. **Regression on the indenture**: with no per-doc config in place, ingest `4. Sixth Street CLO 29 - Indenture (with Final OC) (Executed).pdf` using `_defaults` and confirm section count matches today's output of `_test_step1_parse.py`. Diff old vs new section lists; they should be identical.
3. **Regression on the credit agreement**: copy `configs/sample_credit_agreement.yaml` to `configs/<credit_agreement_doc_id>.yaml`, ingest, confirm section count matches today's output.
4. **End-to-end agentic configure**: run `python -m RAG.configure RAG/Indenture_pdf/<some_new_pdf>` on a doc you have not configured before; watch the streamed node events to confirm the loop iterates (propose → validate → smoke_test → critique) and lands on `accept` within `max_iter` (default 5). Final smoke metrics must satisfy: ≥10 sections, definitions section identified, no single section absorbing >50% of the doc. The accepted YAML's header comment should record iteration count.
5. **Loop safety**: deliberately seed a broken regex in the first proposal (monkey-patch the LLM call) and confirm the graph routes back through `propose_config` with the error visible in state, rather than crashing or silently writing a broken config.
6. **Chunker parity**: run `_inspect_chunks.py` against both docs before and after the refactor; histograms (count by `chunk_type`, count by section_id) should match.
7. **Retrieval smoke**: re-run `_test_step3_retrieve.py` against the re-ingested Chroma collection; top-5 results for a known query should overlap ≥4/5 with the pre-refactor baseline.
