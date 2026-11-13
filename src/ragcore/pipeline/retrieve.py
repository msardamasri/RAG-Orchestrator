"""Retrieve: word search + vector search → Reciprocal Rank Fusion → rerank → the few best chunks.

Word search finds exact terms (codes, ids, names); vector search finds meaning when the words differ. Fusing
ranks instead of scores avoids having to make BM25 and cosine similarity comparable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from ragcore.interfaces import Embedder, Reranker, Store
from ragcore.types import Filters, Hit, RetrievalMode


@dataclass
class RetrievalSettings:
    lexical_k: int = 20
    vector_k: int = 20
    rrf_k: int = 60
    rerank_candidates: int = 30
    top_k: int = 4
    # Evidence floors: the best hit must reach these or the pipeline abstains before generating.
    min_rerank_score: float = 0.0
    min_vector_score: float = 0.0


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hit]], k: int = 60) -> list[Hit]:
    """Merge ranked lists. A chunk's fused score is the sum of ``1 / (k + rank)`` over the lists it appears in."""
    fused: dict[str, Hit] = {}
    totals: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            totals[hit.chunk.id] = totals.get(hit.chunk.id, 0.0) + 1.0 / (k + rank)
            if hit.chunk.id in fused:
                fused[hit.chunk.id].scores.update(hit.scores)
            else:
                fused[hit.chunk.id] = replace(hit, scores=dict(hit.scores))
    ordered = sorted(fused.values(), key=lambda h: totals[h.chunk.id], reverse=True)
    for hit in ordered:
        hit.score = totals[hit.chunk.id]
        hit.scores["rrf"] = hit.score
    return ordered


def retrieve(query: str, *, store: Store, embedder: Embedder, reranker: Reranker | None = None,
             settings: RetrievalSettings | None = None, filters: Filters | None = None,
             mode: RetrievalMode = "hybrid", query_vector: Sequence[float] | None = None) -> list[Hit]:
    settings = settings or RetrievalSettings()
    use_lexical = mode in ("lexical", "hybrid") and store.supports_lexical
    use_vector = mode in ("vector", "hybrid") or not use_lexical

    rankings: list[list[Hit]] = []
    if use_lexical:
        rankings.append(store.search_lexical(query, settings.lexical_k, filters))
    if use_vector:
        vector = query_vector if query_vector is not None else embedder.embed_query(query)
        rankings.append(store.search_vector(vector, settings.vector_k, filters))

    candidates = rankings[0] if len(rankings) == 1 else reciprocal_rank_fusion(rankings, settings.rrf_k)
    if reranker is not None:
        return reranker.rerank(query, candidates[:settings.rerank_candidates], settings.top_k)
    return candidates[:settings.top_k]


def evidence_is_weak(hits: Sequence[Hit], settings: RetrievalSettings) -> bool:
    """Whether the best hit falls under the configured floor for the stage that scored it."""
    if not hits:
        return True
    best = hits[0]
    if "rerank" in best.scores:
        return best.scores["rerank"] < settings.min_rerank_score
    vector_scores = [h.scores["vector"] for h in hits if "vector" in h.scores]
    if vector_scores and settings.min_vector_score > 0:
        return max(vector_scores) < settings.min_vector_score
    return False
