from __future__ import annotations

import hashlib
from typing import TypedDict

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from RAG.pre_injection.config import ParsedConfig
from RAG.pdf_parser import Section


class ChunkMetadata(TypedDict):
    chapter: str | None
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    chunk_type: str
    term: str | None
    doc_id: str


class Chunk(TypedDict):
    id: str
    text: str
    metadata: ChunkMetadata


_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


def chunk_sections(
    sections: list[Section],
    doc_id: str,
    config: ParsedConfig,
    max_tokens: int = 10000,
    chunk_overlap_tokens: int = 200,
    definitions_intro_tokens: int = 500,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(
            _chunk_one_section(
                section,
                doc_id=doc_id,
                config=config,
                max_tokens=max_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                definitions_intro_tokens=definitions_intro_tokens,
            )
        )
    return chunks


def _chunk_one_section(
    section: Section,
    doc_id: str,
    config: ParsedConfig,
    max_tokens: int,
    chunk_overlap_tokens: int,
    definitions_intro_tokens: int,
) -> list[Chunk]:
    text = section["text"]
    if not text.strip():
        return []

    n_tokens = count_tokens(text)
    is_definitions = _is_definitions_section(section, config)

    if n_tokens <= max_tokens and not is_definitions:
        return [_make_chunk(section, text, "whole_section", None, 0, doc_id)]

    if is_definitions:
        term_chunks = _split_definitions(
            section, doc_id, definitions_intro_tokens, config
        )
        if len(term_chunks) >= 5:
            return term_chunks

    return _recursive_subsplit(section, doc_id, max_tokens, chunk_overlap_tokens)


def _is_definitions_section(section: Section, config: ParsedConfig) -> bool:
    defs = config.definitions
    title = section["section_title"].lower()
    for prefix in defs.title_prefix_hints:
        if title.startswith(prefix.lower()):
            return True
    if section["section_id"] in defs.section_id_hints:
        return True
    return False


def _split_definitions(
    section: Section, doc_id: str, intro_tokens: int, config: ParsedConfig
) -> list[Chunk]:
    text = section["text"]
    split_re = config.definitions.term_split_re
    head_re = config.definitions.term_head_re
    parts = split_re.split(text)

    chunks: list[Chunk] = []
    intro_text: str | None = None
    term_blocks: list[tuple[str, str]] = []

    i = 0
    if parts and not head_re.match(parts[0].lstrip()):
        intro_text = parts[0].strip()
        i = 1

    while i < len(parts):
        if i + 1 < len(parts):
            term = parts[i]
            body = parts[i + 1].strip()
            term_blocks.append((term, body))
            i += 2
        else:
            i += 1

    if intro_text:
        intro_truncated = _truncate_to_tokens(intro_text, intro_tokens)
        chunks.append(
            _make_chunk(section, intro_truncated, "whole_section", None, 0, doc_id)
        )

    for idx, (term, body) in enumerate(term_blocks):
        chunks.append(_make_chunk(section, body, "defined_term", term, idx, doc_id))

    return chunks


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    ids = _TOKENIZER.encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        return text
    return _TOKENIZER.decode(ids[:max_tokens])


def _recursive_subsplit(
    section: Section,
    doc_id: str,
    max_tokens: int,
    chunk_overlap_tokens: int,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=max_tokens,
        chunk_overlap=chunk_overlap_tokens,
        separators=["\n\n"],
    )
    pieces = splitter.split_text(section["text"])
    return [
        _make_chunk(section, piece, "sub_split", None, idx, doc_id)
        for idx, piece in enumerate(pieces)
    ]


def _make_chunk(
    section: Section,
    text: str,
    chunk_type: str,
    term: str | None,
    index: int,
    doc_id: str,
) -> Chunk:
    # Always include the index so identically-quoted terms appearing more
    # than once in a section don't collide on chunk_id.
    discriminator = f"{term}|{index}" if term is not None else str(index)
    raw = f"{doc_id}|{section['section_id']}|{chunk_type}|{discriminator}"
    chunk_id = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            chapter=section["chapter"],
            section_id=section["section_id"],
            section_title=section["section_title"],
            page_start=section["page_start"],
            page_end=section["page_end"],
            chunk_type=chunk_type,
            term=term,
            doc_id=doc_id,
        ),
    )
