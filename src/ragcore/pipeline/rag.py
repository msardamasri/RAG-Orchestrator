"""The assembled pipeline. One object a brain holds: ``ingest``, ``retrieve``, ``ask``, ``chat``."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ragcore.interfaces import Chunker, Embedder, Generator, Loader, Reranker, Store
from ragcore.pipeline import answer as answering
from ragcore.pipeline import conversation as conv
from ragcore.pipeline.ingest import IngestReport, ingest
from ragcore.pipeline.retrieve import RetrievalSettings, evidence_is_weak, retrieve
from ragcore.types import Answer, ConversationState, Document, Filters, Hit, RetrievalMode


@dataclass
class Rag:
    store: Store
    embedder: Embedder
    chunker: Chunker
    generator: Generator | None = None
    reranker: Reranker | None = None
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    conversation: conv.ConversationSettings = field(default_factory=conv.ConversationSettings)
    fingerprint: str = ""
    max_tokens: int = 512
    temperature: float = 0.1

    # -- ingest -----------------------------------------------------------------------------------
    def ingest(self, source: Loader | Iterable[Document], prune: bool = False) -> IngestReport:
        """Store what ``source`` yields. A loader with a ``failed`` dict (like ``FileLoader``) has those files
        reported in ``IngestReport.failed`` and protected from ``prune``."""
        documents = source.load() if hasattr(source, "load") else source
        failed = getattr(source, "failed", None)
        if not isinstance(failed, dict):
            failed = {}              # the loader's own dict is kept (not copied): it fills while loading
        report = ingest(documents, chunker=self.chunker, embedder=self.embedder, store=self.store,
                        fingerprint=self.fingerprint, prune=prune, keep=failed)
        report.failed = dict(failed)
        return report

    # -- single question --------------------------------------------------------------------------
    def retrieve(self, query: str, filters: Filters | None = None, mode: RetrievalMode = "hybrid",
                 rerank: bool = True, query_vector: Sequence[float] | None = None) -> list[Hit]:
        """``query_vector``: the query's embedding when the caller already has it (it must come from
        ``embedder.embed_query(query)``)."""
        return retrieve(query, store=self.store, embedder=self.embedder,
                        reranker=self.reranker if rerank else None, settings=self.retrieval, filters=filters,
                        mode=mode, query_vector=query_vector)

    def _require_generator(self) -> Generator:
        if self.generator is None:
            raise RuntimeError("this pipeline has no generator configured; retrieval still works")
        return self.generator

    def ask(self, question: str, filters: Filters | None = None, mode: RetrievalMode = "hybrid") -> Answer:
        hits = self.retrieve(question, filters, mode)
        return answering.answer(question, hits, generator=self._require_generator(),
                                weak_evidence=evidence_is_weak(hits, self.retrieval),
                                max_tokens=self.max_tokens, temperature=self.temperature)

    # -- conversation -----------------------------------------------------------------------------
    def resolve(self, message: str, state: ConversationState) -> conv.Resolved:
        """The standalone question for ``message`` (updates the topic state, not the turns)."""
        return conv.resolve(message, state, generator=self.generator, embed_query=self.embedder.embed_query,
                            settings=self.conversation)

    def chat(self, state: ConversationState, message: str, filters: Filters | None = None) -> Answer:
        """Answer ``message`` in the context of ``state`` and record the turn in it.

        ``filters`` given here become the conversation's retrieval scope until the topic changes.
        """
        generator = self._require_generator()
        resolved = self.resolve(message, state)
        if filters is not None:
            state.filters = dict(filters)
        hits = self.retrieve(resolved.question, state.filters or None, query_vector=resolved.vector)
        result = answering.answer(
            resolved.question, hits, generator=generator, weak_evidence=evidence_is_weak(hits, self.retrieval),
            max_tokens=self.max_tokens, temperature=self.temperature, summary=state.summary, facts=state.facts,
            history=[] if resolved.topic_switched else conv.prompt_history(state, self.conversation))
        result.question = message
        result.standalone_question = resolved.question
        result.topic_switched = resolved.topic_switched
        conv.record_turn(state, message, result.text)
        conv.consolidate(state, generator, self.conversation)
        return result
