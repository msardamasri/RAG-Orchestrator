"""Run the evaluation and report it. Four parts:

1. Retrieval comparison: word search, vector search, hybrid, hybrid + reranker, on the same questions.
2. Abstention before generation: do unanswerable questions score lower than answerable ones, and where to
   put the evidence floor.
3. Conversations: does a follow-up retrieve the right passage raw, rewritten by the model, and with the
   hand-written standalone version; and are topic switches detected.
4. Generation (needs a generator): correct answers, citations pointing at the right passage, and "I don't
   know" on unanswerable questions.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from ragcore.eval.dataset import Conversation, Question
from ragcore.eval.metrics import (
    answer_is_correct,
    best_threshold,
    cites_relevant_source,
    first_relevant_rank,
    score_ranks,
)
from ragcore.pipeline.rag import Rag
from ragcore.pipeline.retrieve import evidence_is_weak
from ragcore.types import ConversationState, Turn

MODES = [("lexical", "lexical", False), ("vector", "vector", False), ("hybrid", "hybrid", False),
         ("hybrid+rerank", "hybrid", True)]


def retrieval_comparison(rag: Rag, questions: Sequence[Question]) -> dict:
    answerable = [q for q in questions if q.answerable]
    kinds = sorted({q.kind for q in answerable if q.kind})
    results: dict[str, dict] = {}
    for name, mode, rerank in MODES:
        if rerank and rag.reranker is None:
            continue
        if mode == "lexical" and not rag.store.supports_lexical:
            continue
        if answerable:      # untimed warm-up: models that load on first use must not count as query time
            rag.retrieve(answerable[0].question, mode=mode, rerank=rerank)
        started = time.perf_counter()
        ranks = {q.id: first_relevant_rank(rag.retrieve(q.question, mode=mode, rerank=rerank), q) for q in answerable}
        elapsed = time.perf_counter() - started
        entry = score_ranks(list(ranks.values())).as_dict()
        entry["ms_per_query"] = round(1000 * elapsed / max(1, len(answerable)), 1)
        entry["by_kind"] = {kind: score_ranks([ranks[q.id] for q in answerable if q.kind == kind]).as_dict()
                            for kind in kinds}
        entry["missed"] = [q.id for q in answerable if ranks[q.id] is None]
        results[name] = entry
    return results


def abstention_before_generation(rag: Rag, questions: Sequence[Question]) -> dict:
    """Score of the best retrieved chunk for answerable vs unanswerable questions, and the floor it suggests."""
    stage = "rerank" if rag.reranker is not None else "vector"
    scores: dict[bool, list[float]] = {True: [], False: []}
    flagged: dict[bool, int] = {True: 0, False: 0}
    for q in questions:
        hits = rag.retrieve(q.question)
        stage_scores = [h.scores[stage] for h in hits if stage in h.scores]
        scores[q.answerable].append(max(stage_scores) if stage_scores else 0.0)
        flagged[q.answerable] += evidence_is_weak(hits, rag.retrieval)
    threshold, accuracy = best_threshold(scores[True], scores[False])
    configured = rag.retrieval.min_rerank_score if stage == "rerank" else rag.retrieval.min_vector_score

    def spread(values: list[float]) -> dict:
        if not values:
            return {}
        ordered = sorted(values)
        return {"min": round(ordered[0], 4), "median": round(ordered[len(ordered) // 2], 4),
                "max": round(ordered[-1], 4)}

    return {
        "stage": stage, "configured_floor": configured,
        "answerable": {"n": len(scores[True]), "wrongly_refused": flagged[True], **spread(scores[True])},
        "unanswerable": {"n": len(scores[False]), "correctly_refused": flagged[False], **spread(scores[False])},
        "suggested_floor": round(threshold, 4), "accuracy_at_suggested_floor": round(accuracy, 4),
    }


def conversation_eval(rag: Rag, conversations: Sequence[Conversation]) -> dict:
    """Follow-up turns only (turn 2 onwards): retrieval with the raw message, the model's rewrite and the gold one."""
    use_model = rag.generator is not None
    ranks: dict[str, list[int | None]] = {"raw": [], "gold_standalone": []}
    if use_model:
        ranks["model_rewrite"] = []
    switch = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    similarities: dict[bool, list[float]] = {True: [], False: []}
    rewrites: list[dict] = []
    for conversation in conversations:
        state = ConversationState()
        for index, turn in enumerate(conversation.turns):
            resolved = rag.resolve(turn.user, state)
            if index > 0:
                if resolved.similarity is not None:
                    similarities[turn.topic_switch].append(resolved.similarity)
                key =("tp" if resolved.topic_switched else "fn") if turn.topic_switch else \
                      ("fp" if resolved.topic_switched else "tn")
                switch[key] += 1
                if turn.answerable:
                    question = turn.as_question(turn.user, f"{conversation.id}:{index}")
                    ranks["raw"].append(first_relevant_rank(rag.retrieve(turn.user), question))
                    ranks["gold_standalone"].append(first_relevant_rank(rag.retrieve(turn.standalone), question))
                    if use_model:
                        rank = first_relevant_rank(rag.retrieve(resolved.question, query_vector=resolved.vector),
                                                   question)
                        ranks["model_rewrite"].append(rank)
                        rewrites.append({"turn": question.id, "user": turn.user, "rewritten": resolved.question,
                                         "gold": turn.standalone, "found": rank is not None})
            # Memory holds what the user said; the gold standalone stands in for the assistant's reply so the
            # rewriter sees realistic context without depending on generation quality.
            state.turns += [Turn("user", turn.user), Turn("assistant", f"(answered: {turn.standalone})")]
    precision = switch["tp"] / (switch["tp"] + switch["fp"]) if switch["tp"] + switch["fp"] else None
    recall = switch["tp"] / (switch["tp"] + switch["fn"]) if switch["tp"] + switch["fn"] else None
    threshold, accuracy = best_threshold(similarities[False], similarities[True])

    def span(values: list[float]) -> list[float]:
        return [round(min(values), 3), round(max(values), 3)] if values else []

    return {
        "follow_up_retrieval": {name: score_ranks(values).as_dict() for name, values in ranks.items()},
        "topic_switch": {**switch, "precision": None if precision is None else round(precision, 4),
                         "recall": None if recall is None else round(recall, 4),
                         "configured_threshold": rag.conversation.topic_switch_threshold,
                         "similarity_same_topic": span(similarities[False]),
                         "similarity_at_switch": span(similarities[True]),
                         "suggested_threshold": round(threshold, 3),
                         "accuracy_at_suggested_threshold": round(accuracy, 4)},
        "rewrites": rewrites,
    }


def generation_eval(rag: Rag, questions: Sequence[Question]) -> dict:
    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]
    rows = []
    started = time.perf_counter()
    for q in questions:
        answer = rag.ask(q.question)
        rows.append({"id": q.id, "answerable": q.answerable, "abstained": answer.abstained, "reason": answer.reason,
                     "correct": q.answerable and answer_is_correct(answer, q),
                     "cites_relevant": q.answerable and cites_relevant_source(answer, q),
                     "answer": answer.text, "raw": answer.raw_text})
    elapsed = time.perf_counter() - started
    by_id = {r["id"]: r for r in rows}

    def rate(items: Sequence[Question], key: str) -> float | None:
        return round(sum(1 for q in items if by_id[q.id][key]) / len(items), 4) if items else None

    answered = [q for q in answerable if not by_id[q.id]["abstained"]]
    return {
        "seconds_per_question": round(elapsed / max(1, len(questions)), 2),
        "answerable": {"n": len(answerable), "answered": len(answered), "correct": rate(answerable, "correct"),
                       "wrongly_refused": rate(answerable, "abstained"),
                       "answered_citing_the_right_passage": rate(answered, "cites_relevant")},
        "unanswerable": {"n": len(unanswerable), "correctly_refused": rate(unanswerable, "abstained")},
        "rows": rows,
    }


def run(rag: Rag, questions: Sequence[Question], conversations: Sequence[Conversation] = (),
        generate: bool = False) -> dict:
    report = {
        "parts": {"embedder": rag.embedder.name, "reranker": getattr(rag.reranker, "name", "none"),
                  "generator": getattr(rag.generator, "name", "none"), "chunks": rag.store.count(),
                  "top_k": rag.retrieval.top_k},
        "retrieval": retrieval_comparison(rag, questions),
        "abstention": abstention_before_generation(rag, questions),
    }
    if conversations:
        report["conversations"] = conversation_eval(rag, conversations)
    if generate:
        if rag.generator is None:
            raise RuntimeError("generation evaluation needs a generator")
        report["generation"] = generation_eval(rag, questions)
    return report


def run_without_generator(rag: Rag, questions: Sequence[Question], conversations: Sequence[Conversation] = ()) -> dict:
    return run(replace(rag, generator=None), questions, conversations, generate=False)


# -- reporting ------------------------------------------------------------------------------------
def pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{100 * value:5.1f}%"


def format_report(report: dict) -> str:
    parts = report["parts"]
    k = parts["top_k"]
    lines = [f"Embedder: {parts['embedder']}   Reranker: {parts['reranker']}   Generator: {parts['generator']}",
             f"Chunks in store: {parts['chunks']}   Kept per question: {k}", "",
             "RETRIEVAL (answerable questions: did the passage holding the answer come back?)",
             f"  {'mode':<16}{'first hit':>10}{'in top ' + str(k):>11}{'MRR':>8}{'ms/query':>10}   missed"]
    for name, entry in report["retrieval"].items():
        lines.append(f"  {name:<16}{pct(entry['hit@1']):>10}{pct(entry['hit@k']):>11}{entry['mrr']:>8.3f}"
                     f"{entry['ms_per_query']:>10.1f}   {', '.join(entry['missed']) or '-'}")
    kinds = sorted({kind for entry in report["retrieval"].values() for kind in entry["by_kind"]})
    if kinds:
        lines += ["", f"  in top {k}, by question type:", f"  {'mode':<16}" + "".join(f"{kind:>14}" for kind in kinds)]
        for name, entry in report["retrieval"].items():
            lines.append(f"  {name:<16}" + "".join(
                f"{pct(entry['by_kind'][kind]['hit@k']):>14}" for kind in kinds))

    abstention = report["abstention"]
    lines += ["", f"REFUSING BEFORE GENERATION (best {abstention['stage']} score per question)",
              f"  answerable    n={abstention['answerable']['n']:<3} scores {abstention['answerable'].get('min')}"
              f" .. {abstention['answerable'].get('max')} (median {abstention['answerable'].get('median')})",
              f"  unanswerable  n={abstention['unanswerable']['n']:<3} scores {abstention['unanswerable'].get('min')}"
              f" .. {abstention['unanswerable'].get('max')} (median {abstention['unanswerable'].get('median')})",
              f"  floor configured: {abstention['configured_floor']} -> refuses "
              f"{abstention['unanswerable']['correctly_refused']}/{abstention['unanswerable']['n']} unanswerable, "
              f"{abstention['answerable']['wrongly_refused']}/{abstention['answerable']['n']} answerable",
              f"  best floor on this data: {abstention['suggested_floor']} "
              f"({pct(abstention['accuracy_at_suggested_floor']).strip()} of questions classified correctly)"]

    if "conversations" in report:
        conversations = report["conversations"]
        lines += ["", "CONVERSATIONS (follow-up turns: passage found in top " + str(k) + ")"]
        for name, entry in conversations["follow_up_retrieval"].items():
            lines.append(f"  {name:<16}{pct(entry['hit@k']):>8}   (n={entry['n']}, MRR {entry['mrr']:.3f})")
        switch = conversations["topic_switch"]
        lines.append(f"  topic switches: caught {switch['tp']} of {switch['tp'] + switch['fn']}, "
                     f"{switch['fp']} false alarms in {switch['fp'] + switch['tn']} same-topic turns")
        lines.append(f"  similarity to the running topic: same topic {switch['similarity_same_topic']}, at a switch "
                     f"{switch['similarity_at_switch']}; threshold {switch['configured_threshold']} configured, "
                     f"{switch['suggested_threshold']} separates best "
                     f"({pct(switch['accuracy_at_suggested_threshold']).strip()})")

    if "generation" in report:
        generation = report["generation"]
        answerable, unanswerable = generation["answerable"], generation["unanswerable"]
        lines += ["", f"GENERATION ({generation['seconds_per_question']} s per question)",
                  f"  answerable:   correct {pct(answerable['correct']).strip()}, wrongly refused "
                  f"{pct(answerable['wrongly_refused']).strip()}, answers citing the right passage "
                  f"{pct(answerable['answered_citing_the_right_passage']).strip()}  (n={answerable['n']})",
                  f"  unanswerable: correctly refused {pct(unanswerable['correctly_refused']).strip()}"
                  f"  (n={unanswerable['n']})"]
    return "\n".join(lines)


def save_report(report: dict, directory: str | Path = "eval_results") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
