"""Rerankers: read each candidate next to the question and reorder. Slower than retrieval, far more precise."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

from ragcore.types import Hit


class NoReranker:
    """Keeps the retrieval order."""

    name = "none"

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        return list(hits[:top_k])


class CrossEncoderReranker:
    """A local cross-encoder through sentence-transformers (extra: ``local``). Scores are 0..1.

    The model loads on the first ``rerank`` call, so commands that never rerank (``ingest``, ``search
    --no-rerank``) do not pay for loading it.
    """

    def __init__(self, model: str = "BAAI/bge-reranker-base", device: str | None = None, batch_size: int = 16,
                 max_length: int = 512):
        self.name = f"cross_encoder:{model}"
        self.model_name = model
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
        return self._model

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        raw = self.model.predict([(query, hit.chunk.text) for hit in hits], batch_size=self.batch_size,
                                 show_progress_bar=False)
        scores = [float(s) for s in raw]
        if any(s < 0.0 or s > 1.0 for s in scores):   # the model returned logits
            scores = [1.0 / (1.0 + math.exp(-s)) for s in scores]
        ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [replace(hit, score=score, scores={**hit.scores, "rerank": score}) for hit, score in ranked[:top_k]]


class CallableReranker:
    """Wrap any ``score(query, text) -> float``. Used by tests, and handy for a brain with its own scorer."""

    name = "callable"

    def __init__(self, score):
        self.score = score

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        scored = [(hit, float(self.score(query, hit.chunk.text))) for hit in hits]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [replace(hit, score=score, scores={**hit.scores, "rerank": score}) for hit, score in scored[:top_k]]
