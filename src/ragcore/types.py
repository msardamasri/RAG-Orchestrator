"""Plain data shapes shared by every part of the library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Filters = dict[str, Any]
"""Metadata equality filters: ``{"source": "a.md"}`` or ``{"tag": ["hr", "legal"]}`` (any of)."""

RetrievalMode = Literal["lexical", "vector", "hybrid"]


@dataclass
class Document:
    """One unit of source material before chunking (a file, a chat, a CRM record...)."""

    id: str
    text: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    """A retrieved chunk. ``score`` is the score of the last stage that ranked it; ``scores`` keeps every stage's.

    Stage keys: ``lexical`` (BM25, higher is better), ``vector`` (cosine similarity), ``rrf``, ``rerank`` (0..1).
    """

    chunk: Chunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class Citation:
    n: int
    chunk_id: str
    doc_id: str
    source: str


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    standalone_question: str | None = None
    reason: str = ""
    """Why it abstained: ``no_hits`` | ``weak_evidence`` | ``model_declined`` | ``no_citations`` | ""."""
    raw_text: str = ""
    topic_switched: bool = False


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class ConversationState:
    """Tiered memory of one conversation: recent turns verbatim, a rolling summary, extracted facts."""

    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    facts: dict[str, str] = field(default_factory=dict)
    filters: Filters = field(default_factory=dict)
    topic_vector: list[float] | None = None
    summarized_upto: int = 0
    """How many of ``turns`` are already folded into ``summary``."""
