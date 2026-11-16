"""Load documents from records a brain already holds: chat messages, CRM rows, notes.

A brain's own loader usually just wraps this: turn its rows into dicts and say which fields are the id,
the text and the metadata.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ragcore.types import Document


class RecordLoader:
    def __init__(self, records: Iterable[Mapping[str, Any]], *, id_field: str = "id", text_field: str = "text",
                 source_field: str | None = None, metadata_fields: Iterable[str] = (),
                 text_fn: Callable[[Mapping[str, Any]], str] | None = None):
        self.records = records
        self.id_field = id_field
        self.text_field = text_field
        self.source_field = source_field
        self.metadata_fields = tuple(metadata_fields)
        self.text_fn = text_fn

    def load(self) -> Iterable[Document]:
        for record in self.records:
            text = self.text_fn(record) if self.text_fn else str(record.get(self.text_field) or "")
            if not text.strip():
                continue
            doc_id = str(record[self.id_field])
            source = str(record.get(self.source_field) or doc_id) if self.source_field else doc_id
            metadata = {k: record[k] for k in self.metadata_fields if record.get(k) is not None}
            yield Document(id=doc_id, text=text, source=source, metadata=metadata)


def conversation_documents(messages: Iterable[Mapping[str, Any]], *, conversation_field: str = "conversation_id",
                           sender_field: str = "sender", text_field: str = "text",
                           window: int = 8, overlap: int = 2) -> Iterable[Document]:
    """Group chat messages into overlapping windows, one document per window, so a reply keeps its question.

    Messages must arrive ordered by time within each conversation.
    """
    by_conversation: dict[str, list[Mapping[str, Any]]] = {}
    for message in messages:
        by_conversation.setdefault(str(message[conversation_field]), []).append(message)
    step = max(1, window - overlap)
    for conversation_id, items in by_conversation.items():
        for start in range(0, len(items), step):
            part = items[start:start + window]
            if not part:
                break
            text = "\n".join(f"{m.get(sender_field, '?')}: {m.get(text_field, '')}" for m in part)
            yield Document(id=f"{conversation_id}:{start}", text=text, source=conversation_id,
                           metadata={"conversation_id": conversation_id, "kind": "chat"})
            if start + window >= len(items):
                break
