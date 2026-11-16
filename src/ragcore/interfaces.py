"""The seams. A new brain swaps any of these without touching the pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from ragcore.types import Chunk, ConversationState, Document, Filters, Hit

Message = dict[str, str]
"""An OpenAI-style chat message: ``{"role": "system" | "user" | "assistant", "content": "..."}``."""


@runtime_checkable
class Loader(Protocol):
    def load(self) -> Iterable[Document]: ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Store(Protocol):
    supports_lexical: bool

    def document_hash(self, doc_id: str) -> str | None:
        """The content hash stored for a document, or None when it was never ingested."""
        ...

    def replace_document(self, document: Document, content_hash: str, chunks: Sequence[Chunk],
                         vectors: Sequence[Sequence[float]]) -> None:
        """Store a document's chunks, removing whatever chunks it had before."""
        ...

    def delete_document(self, doc_id: str) -> None: ...
    def search_vector(self, vector: Sequence[float], k: int, filters: Filters | None = None) -> list[Hit]: ...
    def search_lexical(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]: ...
    def count(self) -> int: ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]: ...


@runtime_checkable
class Generator(Protocol):
    name: str

    def complete(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.1) -> str: ...


@runtime_checkable
class MemoryStore(Protocol):
    def load(self, session_id: str) -> ConversationState: ...
    def save(self, session_id: str, state: ConversationState) -> None: ...
