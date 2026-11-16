"""Chunk by section, paragraph and sentence, never mid-sentence, with overlap between neighbours.

Markdown headings start a new chunk and travel with it (``Title > Section`` is the chunk's first line), so a
chunk keeps the context that says what it is about. Lines inside fenced code blocks are never headings. Sizes
are in characters.
"""

from __future__ import annotations

import re
import uuid

from ragcore.types import Chunk, Document

# Up to three spaces of indent, then 1-6 '#'; closing '#'s are not part of the title (CommonMark).
HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
# A sentence ends at . ! or ? followed by a capital, a digit or an opening mark, except after a list or section
# number at the start of a line ("1. Scope", "12. Exceptions"), which starts a sentence instead of ending one.
SENTENCE_END = re.compile(
    r"(?<=[.!?])(?<!^\d\.)(?<!^\d\d\.)(?<!\n\d\.)(?<!\n\d\d\.)\s+(?=[A-ZÁÉÍÓÚÑ¿¡\"'(\[0-9])")
# Where a unit that is still too long may be cut, best first: a line break, a clause, a word.
SPLIT_POINTS = ("\n", "; ", ", ", " ")
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

    fence = ""                                   # the marker that opened the current code block, if any
    for line in text.splitlines():
        opener = FENCE.match(line)
        if fence:
            body.append(line)
            # A fence closes with the same character, at least as long, and nothing else on the line.
            if opener and opener.group(1)[0] == fence[0] and len(opener.group(1)) >= len(fence) \
                    and not line.strip().lstrip(fence[0]):
                fence = ""
            continue
        if opener:
            fence = opener.group(1)
            body.append(line)
            continue
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


def split_long(text: str, size: int, points: tuple[str, ...] = SPLIT_POINTS) -> list[str]:
    """Pieces of at most ``size`` characters, cut at the best split point available.

    Lines become separate pieces, so table rows and list items stay whole and the chunker can still overlap
    them. A single line that is too long is cut at clauses, then words, and only cut mid-word when one word
    alone is longer than ``size``.
    """
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    if not points:
        return [text[i:i + size] for i in range(0, len(text), size)]
    point, finer = points[0], points[1:]
    if point not in text:
        return split_long(text, size, finer)
    if point == "\n":
        return [piece for line in text.split("\n") for piece in split_long(line, size, finer)]
    trail = point.rstrip()                      # the "," or ";" a piece keeps where it is cut
    pieces: list[str] = []
    current = ""
    for part in text.split(point):
        candidate = f"{current}{point}{part}" if current else part
        if len(candidate) + len(trail) <= size:
            current = candidate
            continue
        if current:
            pieces.append(current.strip() + trail)
        current = ""
        if len(part) + len(trail) <= size:
            current = part
        else:
            pieces.extend(split_long(part, size, finer))
    if current.strip():
        pieces.append(current.strip())
    return pieces


def _units(body: str, size: int) -> list[str]:
    """Paragraphs, split into sentences only when a paragraph alone exceeds ``size``, and a sentence that is
    still too long (a table, a list without full stops) split by ``split_long``."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= size:
            units.append(paragraph)
            continue
        for sentence in split_sentences(paragraph):
            units.extend(split_long(sentence, size))
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
