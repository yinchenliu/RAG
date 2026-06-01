from __future__ import annotations

import re
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def _validate_regex(pattern: str) -> str:
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValueError(f"invalid regex {pattern!r}: {e}") from e
    return pattern


def _require_groups(pattern: str, min_groups: int, role: str) -> str:
    """Compile + require at least `min_groups` capturing groups. Without
    enough groups the parser would call m.group(N) and IndexError out, so
    we surface this at validation time as a clear LLM-fixable error."""
    _validate_regex(pattern)
    n = re.compile(pattern).groups
    if n < min_groups:
        raise ValueError(
            f"{role}: regex {pattern!r} has {n} capturing group(s); need "
            f"at least {min_groups}. Wrap the value(s) you want to extract "
            f"in parentheses, e.g. (\\d+) instead of \\d+."
        )
    return pattern


class TocConfig(BaseModel): 
    heading_pattern: str
    entry_patterns: list[str]
    chapter_pattern: str
    search_pages: int = 20
    blank_page_streak_stop: int = 2
    min_sections_to_accept: int = 5

    @field_validator("heading_pattern")
    @classmethod
    def _v_heading(cls, v: str) -> str:
        return _validate_regex(v)

    @field_validator("chapter_pattern")
    @classmethod
    def _v_chapter(cls, v: str) -> str:
        return _require_groups(v, 3, "toc.chapter_pattern (chapter_num, title, page_num)")

    @field_validator("entry_patterns")
    @classmethod
    def _v_entries(cls, v: list[str]) -> list[str]:
        for p in v:
            _require_groups(p, 3, "toc.entry_patterns[*] (section_id, title, page_num)")
        return v

    @cached_property
    def heading_re(self) -> re.Pattern[str]:
        return re.compile(self.heading_pattern)

    @cached_property
    def chapter_re(self) -> re.Pattern[str]:
        return re.compile(self.chapter_pattern)

    @cached_property
    def entry_res(self) -> list[re.Pattern[str]]:
        return [re.compile(p) for p in self.entry_patterns]


class BodyConfig(BaseModel):
    section_header_pattern: str
    section_id_only_pattern: str
    chapter_pattern: str
    chapter_allcaps_pattern: str
    schedule_cutoff_pattern: str
    body_start_markers: list[str] = Field(default_factory=list)
    line_skip_patterns: list[str] = Field(default_factory=list)
    page_skip_after_toc: int = 10

    @field_validator("section_header_pattern")
    @classmethod
    def _v_section_header(cls, v: str) -> str:
        return _require_groups(v, 2, "body.section_header_pattern (section_id, title)")

    @field_validator("section_id_only_pattern")
    @classmethod
    def _v_section_id_only(cls, v: str) -> str:
        return _require_groups(v, 1, "body.section_id_only_pattern (section_id)")

    @field_validator("chapter_pattern")
    @classmethod
    def _v_chapter(cls, v: str) -> str:
        return _require_groups(v, 1, "body.chapter_pattern (chapter_num)")

    @field_validator("chapter_allcaps_pattern")
    @classmethod
    def _v_chapter_allcaps(cls, v: str) -> str:
        return _require_groups(v, 2, "body.chapter_allcaps_pattern (chapter_num, title)")

    @field_validator("schedule_cutoff_pattern")
    @classmethod
    def _v_schedule(cls, v: str) -> str:
        return _validate_regex(v)

    @field_validator("line_skip_patterns")
    @classmethod
    def _v_skips(cls, v: list[str]) -> list[str]:
        for p in v:
            _validate_regex(p)
        return v

    @cached_property
    def section_header_re(self) -> re.Pattern[str]:
        return re.compile(self.section_header_pattern)

    @cached_property
    def section_id_only_re(self) -> re.Pattern[str]:
        return re.compile(self.section_id_only_pattern)

    @cached_property
    def chapter_re(self) -> re.Pattern[str]:
        return re.compile(self.chapter_pattern)

    @cached_property
    def chapter_allcaps_re(self) -> re.Pattern[str]:
        return re.compile(self.chapter_allcaps_pattern)

    @cached_property
    def schedule_cutoff_re(self) -> re.Pattern[str]:
        return re.compile(self.schedule_cutoff_pattern)

    @cached_property
    def line_skip_res(self) -> list[re.Pattern[str]]:
        return [re.compile(p) for p in self.line_skip_patterns]


class PageNumberingConfig(BaseModel):
    printed_page_pattern: str
    search_first_n_lines: int = 3
    search_last_n_lines: int = 3

    @field_validator("printed_page_pattern")
    @classmethod
    def _v_regex(cls, v: str) -> str:
        return _require_groups(v, 1, "page_numbering.printed_page_pattern (page_num)")

    @cached_property
    def printed_page_re(self) -> re.Pattern[str]:
        return re.compile(self.printed_page_pattern)


class HeaderFooterStripConfig(BaseModel):
    enable: bool = True
    min_span_pages: int = 4
    min_line_length: int = 3
    max_line_length: int = 120
    min_repetition_fraction: float = 0.5


class DefinitionsConfig(BaseModel):
    section_id_hints: list[str] = Field(default_factory=list)
    title_prefix_hints: list[str] = Field(default_factory=list)
    term_split_pattern: str
    term_head_pattern: str

    @field_validator("term_split_pattern", "term_head_pattern")
    @classmethod
    def _v_regex(cls, v: str) -> str:
        return _require_groups(v, 1, "definitions.term_split_pattern/term_head_pattern (term_name)")

    @cached_property
    def term_split_re(self) -> re.Pattern[str]:
        return re.compile(self.term_split_pattern)

    @cached_property
    def term_head_re(self) -> re.Pattern[str]:
        return re.compile(self.term_head_pattern)


class ParsedConfig(BaseModel):
    schema_version: int = 1
    toc: TocConfig
    body: BodyConfig
    page_numbering: PageNumberingConfig
    header_footer_strip: HeaderFooterStripConfig
    definitions: DefinitionsConfig

    model_config = {"arbitrary_types_allowed": True}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config_path(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    candidate = _CONFIGS_DIR / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"config {name_or_path!r} not found (looked at {p} and {candidate})"
    )


def _load_chain(name_or_path: str | Path, seen: set[Path] | None = None) -> dict[str, Any]:
    seen = seen if seen is not None else set()
    path = _resolve_config_path(name_or_path)
    if path in seen:
        raise ValueError(f"circular extends chain at {path}")
    seen.add(path)
    data = _read_yaml(path)
    parent_name = data.pop("extends", None)
    if parent_name is None:
        return data
    parent = _load_chain(parent_name, seen)
    return _deep_merge(parent, data)


def load_config(name_or_path: str | Path = "_defaults") -> ParsedConfig:
    """Load and validate a parsing/chunking config.

    Resolution: if `name_or_path` is a `.yaml`/`.yml` file that exists, load
    it; otherwise look up `RAG/pre_injection/configs/<name>.yaml`. Handles `extends:` chains
    by deep-merging the parent config into the child. Every regex field is
    validated at load time, so a bad pattern fails here instead of mid-parse.
    """
    data = _load_chain(name_or_path)
    return ParsedConfig.model_validate(data)
