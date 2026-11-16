"""Metrics that need no judge model: did the right passage come back, and how high."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ragcore.eval.dataset import Question
from ragcore.types import Answer, Hit


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def is_relevant(hit: Hit, question: Question) -> bool:
    """A hit is relevant when it comes from an expected document and contains one of the expected fragments."""
    if question.expected_docs and hit.chunk.doc_id not in question.expected_docs:
        return False
    if not question.expected_texts:
        return bool(question.expected_docs)
    text = normalize(hit.chunk.text)
    return any(normalize(fragment) in text for fragment in question.expected_texts)


def first_relevant_rank(hits: Sequence[Hit], question: Question) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit, question):
            return rank
    return None


@dataclass
class RetrievalScore:
    n: int = 0
    hit_at_1: float = 0.0
    hit_at_k: float = 0.0
    mrr: float = 0.0

    def as_dict(self) -> dict:
        return {"n": self.n, "hit@1": round(self.hit_at_1, 4), "hit@k": round(self.hit_at_k, 4),
                "mrr": round(self.mrr, 4)}


def score_ranks(ranks: Sequence[int | None]) -> RetrievalScore:
    n = len(ranks)
    if not n:
        return RetrievalScore()
    return RetrievalScore(n=n, hit_at_1=sum(1 for r in ranks if r == 1) / n,
                          hit_at_k=sum(1 for r in ranks if r is not None) / n,
                          mrr=sum(1.0 / r for r in ranks if r is not None) / n)


def best_threshold(positives: Sequence[float], negatives: Sequence[float]) -> tuple[float, float]:
    """The score floor that best separates answerable (positives) from unanswerable (negatives) questions.

    Returns ``(threshold, accuracy)``: questions scoring below the threshold are treated as unanswerable.
    """
    if not positives or not negatives:
        return 0.0, 1.0
    values = sorted(set(positives) | set(negatives))
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(values, values[1:], strict=False)]
    total = len(positives) + len(negatives)
    best = (0.0, 0.0)
    for threshold in candidates:
        correct = sum(1 for p in positives if p >= threshold) + sum(1 for n in negatives if n < threshold)
        if correct / total > best[1]:
            best = (threshold, correct / total)
    return best


def answer_is_correct(answer: Answer, question: Question) -> bool:
    """A cheap correctness check: the answer was given and contains every expected string."""
    if answer.abstained:
        return False
    text = normalize(answer.text)
    return all(normalize(expected) in text for expected in question.answer_contains)


def cites_relevant_source(answer: Answer, question: Question) -> bool:
    """Whether at least one citation points at a passage that actually holds the answer."""
    cited = {c.chunk_id for c in answer.citations}
    return any(h.chunk.id in cited and is_relevant(h, question) for h in answer.hits)
