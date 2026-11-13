"""Chunk by section, paragraph and sentence, never mid-sentence, with overlap between neighbours.

Markdown headings start a new chunk and travel with it (``Title > Section`` is the chunk's first line), so a
chunk keeps the context that says what it is about. Sizes are in characters.
"""

from __future__ import annotations

import re
import uuid

from ragcore.types import Chunk, Document

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡\"'(\[0-9])")
QA_START = re.compile(r"^\s*(?:Q|Question|P|Pregunta)\s*[:.]\s*", re.IGNORECASE)


def chunk_id(doc_id: str, position: int) -> str:
    """Stable across runs, so re-ingesting a document updates its chunks instead of duplicating them."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{position}"))


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(paragraph.strip()) if s.strip()]


def _sections(text: str) -> list[tuple[str, str]]:
    """``[(heading path, body)]``. Text without headings is one section with an empty path."""
    sections: list[tuple[str, str]] = []
    path: list[tuple[int, str]] = []
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append((" > ".join(title for _, title in path), content))
        body.clear()

    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, match.group(2)))
        else:
            body.append(line)
    flush()
    return sections


def _units(body: str, size: int) -> list[str]:
    """Paragraphs, split further into sentences (then hard-cut) only when a paragraph alone exceeds ``size``."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= size:
            units.append(paragraph)
            continue
        for sentence in split_sentences(paragraph):
            while len(sentence) > size:
                units.append(sentence[:size])
                sentence = sentence[size:]
            if sentence:
                units.append(sentence)
    return units


class SentenceChunker:
    def __init__(self, size: int = 1000, overlap: int = 200):
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    def _pack(self, units: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        length = 0
        for unit in units:
            if current and length + len(unit) + 1 > self.size:
                chunks.append("\n".join(current))
                tail: list[str] = []
                tail_length = 0
                for previous in reversed(current):
                    if tail_length + len(previous) > self.overlap:
                        break
                    tail.insert(0, previous)
                    tail_length += len(previous) + 1
                current, length = tail, tail_length
            current.append(unit)
            length += len(unit) + 1
        if current:
            chunks.append("\n".join(current))
        return chunks

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for heading, body in _sections(document.text):
            for text in self._pack(_units(body, self.size)):
                position = len(chunks)
                metadata = {**document.metadata, "source": document.source or document.id}
                if heading:
                    metadata["section"] = heading
                chunks.append(Chunk(id=chunk_id(document.id, position), doc_id=document.id,
                                    text=f"{heading}\n{text}" if heading else text, position=position,
                                    metadata=metadata))
        return chunks


class QAPairChunker:
    """One chunk per question-and-answer pair, for FAQs. Falls back to ``SentenceChunker`` when there are none."""

    def __init__(self, size: int = 1000, overlap: int = 200):
        self.fallback = SentenceChunker(size, overlap)

    def chunk(self, document: Document) -> list[Chunk]:
        pairs: list[list[str]] = []
        for line in document.text.splitlines():
            if QA_START.match(line):
                pairs.append([line.strip()])
            elif pairs and line.strip():
                pairs[-1].append(line.strip())
        if len(pairs) < 2:
            return self.fallback.chunk(document)
        metadata = {**document.metadata, "source": document.source or document.id}
        return [Chunk(id=chunk_id(document.id, i), doc_id=document.id, text="\n".join(pair), position=i,
                      metadata=dict(metadata)) for i, pair in enumerate(pairs)]


class WholeDocumentChunker:
    """One chunk per document: for records that are already the right size (a chat window, a short note)."""

    def chunk(self, document: Document) -> list[Chunk]:
        metadata = {**document.metadata, "source": document.source or document.id}
        return [Chunk(id=chunk_id(document.id, 0), doc_id=document.id, text=document.text, position=0,
                      metadata=metadata)]
