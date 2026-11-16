"""Local embeddings through sentence-transformers (extra: ``local``). Nothing leaves the machine."""

from __future__ import annotations

from collections.abc import Sequence

# Models trained with instruction prefixes. Without them their retrieval quality drops noticeably.
PREFIXES: dict[str, tuple[str, str]] = {
    "e5": ("query: ", "passage: "),
    "bge-": ("Represent this sentence for searching relevant passages: ", ""),
}


def prefixes_for(model: str) -> tuple[str, str]:
    lowered = model.lower()
    if "bge-m3" in lowered:
        return "", ""
    for marker, pair in PREFIXES.items():
        if marker in lowered:
            return pair
    return "", ""


class LocalEmbedder:
    def __init__(self, model: str = "intfloat/multilingual-e5-small", device: str | None = None,
                 batch_size: int = 32, query_prefix: str | None = None, document_prefix: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = f"local:{model}"
        self.model = SentenceTransformer(model, device=device)
        self.batch_size = batch_size
        default_query, default_document = prefixes_for(model)
        self.query_prefix = default_query if query_prefix is None else query_prefix
        self.document_prefix = default_document if document_prefix is None else document_prefix
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                    show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]
