from __future__ import annotations

import pytest

from ragcore.generators.generators import FakeGenerator
from ragcore.pipeline.answer import IDK, build_messages, check_answer
from ragcore.pipeline.retrieve import RetrievalSettings, evidence_is_weak, reciprocal_rank_fusion
from ragcore.rerankers.rerankers import CallableReranker
from ragcore.types import Chunk, Hit


def hit(chunk_id: str, **scores) -> Hit:
    return Hit(chunk=Chunk(id=chunk_id, doc_id=f"{chunk_id}.md", text=f"text {chunk_id}",
                           metadata={"source": f"{chunk_id}.md"}), score=0.0, scores=dict(scores))


def test_rrf_rewards_agreement_between_lists():
    lexical = [hit("a", lexical=9.0), hit("b", lexical=5.0), hit("c", lexical=1.0)]
    vector = [hit("b", vector=0.9), hit("d", vector=0.8), hit("a", vector=0.7)]
    fused = reciprocal_rank_fusion([lexical, vector], k=60)
    assert [h.chunk.id for h in fused] == ["b", "a", "d", "c"]
    assert fused[0].scores == {"lexical": 5.0, "vector": 0.9, "rrf": pytest.approx(1 / 62 + 1 / 61)}
    assert fused[0].score == fused[0].scores["rrf"]
    assert lexical[1].scores == {"lexical": 5.0}            # inputs are not mutated


def test_hybrid_finds_what_each_leg_alone_misses(rag):
    # An exact code: word search nails it.
    assert "EXP-204" in rag.retrieve("EXP-204", mode="lexical")[0].chunk.text
    assert "EXP-204" in rag.retrieve("what is EXP-204", mode="hybrid")[0].chunk.text
    # No word in common with the store: word search returns nothing, the pipeline still answers from vectors.
    assert rag.retrieve("zzzz qqqq", mode="lexical") == []
    assert len(rag.retrieve("zzzz qqqq", mode="hybrid")) == rag.retrieval.top_k


def test_modes_and_top_k(rag):
    rag.retrieval = RetrievalSettings(top_k=2)
    for mode in ("lexical", "vector", "hybrid"):
        hits = rag.retrieve("annual leave days carry over", mode=mode)
        assert len(hits) <= 2 and hits[0].chunk.doc_id == "leave.md"
    assert "rrf" in rag.retrieve("annual leave days", mode="hybrid")[0].scores
    assert "rrf" not in rag.retrieve("annual leave days", mode="vector")[0].scores


def test_reranker_reorders_and_can_be_skipped(rag):
    rag.reranker = CallableReranker(lambda query, text: 1.0 if "Passwords" in text else 0.1)
    assert "Passwords" in rag.retrieve("annual leave days")[0].chunk.text
    assert rag.retrieve("annual leave days")[0].scores["rerank"] == 1.0
    assert rag.retrieve("annual leave days", rerank=False)[0].chunk.doc_id == "leave.md"


def test_filters_scope_retrieval(rag):
    hits = rag.retrieve("receipt", filters={"team": "it"})
    assert hits and {h.chunk.doc_id for h in hits} == {"security.md"}


def test_evidence_floor():
    settings = RetrievalSettings(min_rerank_score=0.3, min_vector_score=0.5)
    assert evidence_is_weak([], settings)
    assert evidence_is_weak([hit("a", rerank=0.2, vector=0.99)], settings)
    assert not evidence_is_weak([hit("a", rerank=0.31)], settings)
    assert evidence_is_weak([hit("a", vector=0.4, rrf=0.03)], settings)
    assert not evidence_is_weak([hit("a", vector=0.4), hit("b", vector=0.6)], settings)
    assert not evidence_is_weak([hit("a", lexical=3.0)], settings)      # no floor defined for word search alone


def test_prompt_numbers_sources_and_carries_memory():
    messages = build_messages("How many days?", [hit("a"), hit("b")], summary="Asked about leave.",
                              facts={"office": "Barcelona"}, history=[{"role": "user", "content": "hi"}])
    assert messages[0]["role"] == "system" and "ONLY the numbered sources" in messages[0]["content"]
    assert "Asked about leave." in messages[0]["content"] and "office = Barcelona" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[-1]["content"] == "Sources:\n[1] (a.md) text a\n[2] (b.md) text b\n\nQuestion: How many days?"


def test_citations_are_checked_against_supplied_sources():
    hits = [hit("a"), hit("b")]
    text, citations, reason = check_answer("It is 25 days [1]. Carry-over is 5 [2, 7] and more [9].", hits)
    assert reason == "" and [c.n for c in citations] == [1, 2]
    assert text == "It is 25 days [1]. Carry-over is 5 [2] and more."
    assert citations[1].doc_id == "b.md" and citations[1].source == "b.md"

    assert check_answer("It is 25 days.", hits) == (IDK, [], "no_citations")
    assert check_answer("It is 25 days [5].", hits) == (IDK, [], "no_citations")
    assert check_answer("I don't know.", hits) == (IDK, [], "model_declined")
    assert check_answer("The sources do not mention this.", hits)[2] == "model_declined"
    assert check_answer("No lo sé.", hits)[2] == "model_declined"
    # A partial answer that cites what it does know is kept.
    assert check_answer("The cap is 5 days [1], but I don't know the deadline.", hits)[2] == ""


def test_ask_answers_with_citations(rag):
    result = rag.ask("How many days of annual leave do employees receive?")
    assert not result.abstained and result.citations[0].n == 1
    assert result.citations[0].doc_id == result.hits[0].chunk.doc_id == "leave.md"
    assert "25 days" in result.text


def test_ask_abstains_without_calling_the_model_when_evidence_is_weak(rag):
    generator = FakeGenerator()
    rag.generator = generator
    rag.retrieval = RetrievalSettings(min_vector_score=0.99)
    result = rag.ask("What is the stock option vesting schedule?", mode="vector")
    assert result.abstained and result.reason == "weak_evidence" and result.text == IDK
    assert generator.calls == []


def test_ask_abstains_when_the_model_does_not_cite(rag):
    rag.generator = FakeGenerator(lambda messages: "Employees get 40 days, trust me.")
    result = rag.ask("How many days of annual leave?")
    assert result.abstained and result.reason == "no_citations" and result.text == IDK
    assert result.raw_text == "Employees get 40 days, trust me."


def test_ask_needs_a_generator(rag):
    rag.generator = None
    assert rag.retrieve("annual leave")
    with pytest.raises(RuntimeError, match="no generator"):
        rag.ask("annual leave?")
