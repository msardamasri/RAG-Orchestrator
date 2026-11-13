"""Ingest: load → chunk → embed → store. Safe to run again: unchanged documents are skipped, changed ones replaced."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field

from ragcore.interfaces import Chunker, Embedder, Store
from ragcore.types import Document


@dataclass
class IngestReport:
    documents: int = 0
    ingested: int = 0
    skipped: int = 0
    removed: int = 0
    chunks: int = 0
    empty: list[str] = field(default_factory=list)


def content_hash(document: Document, fingerprint: str) -> str:
    """Changes when the text, the metadata, or how it is chunked and embedded changes."""
    payload = json.dumps([document.text, document.source, document.metadata, fingerprint], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def ingest(documents: Iterable[Document], *, chunker: Chunker, embedder: Embedder, store: Store,
           fingerprint: str = "", prune: bool = False) -> IngestReport:
    """Store ``documents``. With ``prune``, documents the store holds that are not in this run are removed
    (use it when the run covers the whole corpus, never for a partial update)."""
    report = IngestReport()
    seen: set[str] = set()
    for document in documents:
        report.documents += 1
        seen.add(document.id)
        digest = content_hash(document, f"{fingerprint}|{embedder.name}")
        if store.document_hash(document.id) == digest:
            report.skipped += 1
            continue
        chunks = chunker.chunk(document)
        if not chunks:
            report.empty.append(document.id)
            store.delete_document(document.id)
            continue
        vectors = embedder.embed_documents([c.text for c in chunks])
        store.replace_document(document, digest, chunks, vectors)
        report.ingested += 1
        report.chunks += len(chunks)
    if prune and hasattr(store, "document_ids"):
        for doc_id in store.document_ids():
            if doc_id not in seen:
                store.delete_document(doc_id)
                report.removed += 1
    return report
