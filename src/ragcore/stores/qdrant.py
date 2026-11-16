"""Qdrant store (extra: ``qdrant``): the original project's vector database, kept as an option.

Vector search only. The retriever sees ``supports_lexical = False`` and skips the word-search leg, so a
profile using Qdrant runs vector retrieval (plus the reranker, if one is configured).
"""

from __future__ import annotations

from collections.abc import Sequence

from ragcore.types import Chunk, Document, Filters, Hit


class QdrantStore:
    supports_lexical = False

    def __init__(self, url: str = "http://localhost:6333", collection: str = "documents", dim: int = 3072,
                 api_key: str | None = None, client=None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = client or QdrantClient(url=url, api_key=api_key, timeout=30)
        self.collection = collection
        self.dim = dim
        if not self.client.collection_exists(collection):
            self.client.create_collection(collection_name=collection,
                                          vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    @staticmethod
    def _filter(filters: Filters | None):
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            field = "doc_id" if key == "doc_id" else f"metadata.{key}"
            if isinstance(value, (list, tuple, set)):
                conditions.append(FieldCondition(key=field, match=MatchAny(any=list(value))))
            else:
                conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
        return Filter(must=conditions)

    def document_hash(self, doc_id: str) -> str | None:
        points, _ = self.client.scroll(collection_name=self.collection, scroll_filter=self._filter({"doc_id": doc_id}),
                                       limit=1, with_payload=True)
        return (points[0].payload or {}).get("doc_hash") if points else None

    def delete_document(self, doc_id: str) -> None:
        from qdrant_client.models import FilterSelector

        self.client.delete(collection_name=self.collection,
                           points_selector=FilterSelector(filter=self._filter({"doc_id": doc_id})))

    def replace_document(self, document: Document, content_hash: str, chunks: Sequence[Chunk],
                         vectors: Sequence[Sequence[float]]) -> None:
        from qdrant_client.models import PointStruct

        self.delete_document(document.id)
        points = [
            PointStruct(id=chunk.id, vector=list(vector),
                        payload={"doc_id": chunk.doc_id, "doc_hash": content_hash, "position": chunk.position,
                                 "text": chunk.text, "source": document.source, "metadata": chunk.metadata})
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(self.collection, points=points)

    def search_vector(self, vector: Sequence[float], k: int, filters: Filters | None = None) -> list[Hit]:
        points = self.client.query_points(collection_name=self.collection, query=list(vector), limit=k,
                                          with_payload=True, query_filter=self._filter(filters)).points
        hits = []
        for point in points:
            payload = point.payload or {}
            chunk = Chunk(id=str(point.id), doc_id=payload.get("doc_id", ""), text=payload.get("text", ""),
                          position=int(payload.get("position", 0)), metadata=payload.get("metadata") or {})
            hits.append(Hit(chunk=chunk, score=float(point.score), scores={"vector": float(point.score)}))
        return hits

    def search_lexical(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        return []

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)
