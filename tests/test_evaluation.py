from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragcore import cli
from ragcore.config import RagConfig, build_rag, load_profile
from ragcore.eval.dataset import Conversation, ConversationTurn, Question, load_conversations, load_questions
from ragcore.eval.metrics import (
    answer_is_correct,
    best_threshold,
    cites_relevant_source,
    first_relevant_rank,
    is_relevant,
    score_ranks,
)
from ragcore.eval.runner import format_report, run
from ragcore.stores.sqlite import fts_query
from ragcore.types import Answer, Chunk, Citation, Hit

ROOT = Path(__file__).resolve().parents[1]


def hit(doc: str, text: str, chunk_id: str = "c") -> Hit:
    return Hit(chunk=Chunk(id=chunk_id, doc_id=doc, text=text), score=1.0)


def test_relevance_needs_the_document_and_the_fragment():
    question = Question(id="q", question="?", expected_docs=["leave.md"], expected_texts=["25 days of  Annual leave"])
    assert is_relevant(hit("leave.md", "Employees receive 25 DAYS of annual\nleave per year."), question)
    assert not is_relevant(hit("other.md", "25 days of annual leave"), question)
    assert not is_relevant(hit("leave.md", "Submit form HR-17."), question)
    ranks = [first_relevant_rank([hit("leave.md", "x"), hit("leave.md", "25 days of annual leave")], question)]
    assert ranks == [2]


def test_rank_scores():
    score = score_ranks([1, 2, None, 4])
    assert (score.n, score.hit_at_1, score.hit_at_k) == (4, 0.25, 0.75)
    assert score.mrr == pytest.approx((1 + 0.5 + 0.25) / 4)
    assert score_ranks([]).n == 0


def test_best_threshold_separates_two_groups():
    threshold, accuracy = best_threshold([0.8, 0.9, 0.7], [0.1, 0.2])
    assert 0.2 < threshold < 0.7 and accuracy == 1.0
    assert best_threshold([0.5], []) == (0.0, 1.0)


def test_answer_checks():
    question = Question(id="q", question="?", expected_docs=["leave.md"], expected_texts=["25 days"],
                        answer_contains=["25"])
    hits = [hit("leave.md", "25 days", "good"), hit("leave.md", "form", "bad")]
    good = Answer(question="?", text="You get 25 days [1].", abstained=False, hits=hits,
                  citations=[Citation(1, "good", "leave.md", "leave.md")])
    wrong_source = Answer(question="?", text="You get 25 days [2].", abstained=False, hits=hits,
                          citations=[Citation(2, "bad", "leave.md", "leave.md")])
    refused = Answer(question="?", text="I don't know.", abstained=True, hits=hits)
    assert answer_is_correct(good, question) and cites_relevant_source(good, question)
    assert answer_is_correct(wrong_source, question) and not cites_relevant_source(wrong_source, question)
    assert not answer_is_correct(refused, question)


def test_stopwords_are_dropped_from_word_search_unless_nothing_else_is_left():
    assert fts_query("What is the limit for a taxi?") == '"limit" OR "taxi"'
    assert fts_query("¿Cuántos días de vacaciones hay?") == '"días" OR "vacaciones"'
    assert fts_query("what is it") == '"what" OR "is" OR "it"'


def test_run_reports_every_part_offline(rag):
    questions = [
        Question(id="q1", question="How many days of annual leave do employees receive?", kind="paraphrase",
                 expected_docs=["leave.md"], expected_texts=["25 days of annual leave"], answer_contains=["25"]),
        Question(id="q2", question="What does EXP-204 mean?", kind="exact_term", expected_docs=["expenses.md"],
                 expected_texts=["EXP-204 means the receipt is missing"], answer_contains=["receipt"]),
        Question(id="q3", question="Who is the chief executive?", answerable=False, kind="unanswerable"),
    ]
    conversations = [Conversation(id="c1", turns=[
        ConversationTurn(user="How many days of annual leave do employees receive?",
                         standalone="How many days of annual leave do employees receive?",
                         expected_docs=["leave.md"], expected_texts=["25 days"]),
        ConversationTurn(user="Which form?", standalone="Which form do I submit when requesting leave?",
                         expected_docs=["leave.md"], expected_texts=["HR-17"]),
        ConversationTurn(user="By the way, what does EXP-204 mean?", standalone="What does EXP-204 mean?",
                         topic_switch=True, expected_docs=["expenses.md"], expected_texts=["EXP-204"]),
    ])]
    report = run(rag, questions, conversations, generate=True)
    assert set(report["retrieval"]) == {"lexical", "vector", "hybrid"}          # no reranker configured
    assert report["retrieval"]["hybrid"]["n"] == 2 and report["retrieval"]["hybrid"]["hit@k"] == 1.0
    assert report["retrieval"]["lexical"]["by_kind"]["exact_term"]["hit@1"] == 1.0
    assert report["abstention"]["unanswerable"]["n"] == 1
    switch = report["conversations"]["topic_switch"]
    assert (switch["tp"], switch["fn"], switch["fp"]) == (1, 0, 0)
    assert report["conversations"]["follow_up_retrieval"]["gold_standalone"]["hit@k"] == 1.0
    assert report["generation"]["answerable"]["correct"] == 1.0
    text = format_report(report)
    for heading in ("RETRIEVAL", "REFUSING BEFORE GENERATION", "CONVERSATIONS", "GENERATION"):
        assert heading in text
    json.dumps(report)                                                            # the report is serialisable


def test_bundled_dataset_loads_and_matches_the_corpus():
    questions = load_questions(ROOT / "eval_data" / "questions.jsonl")
    conversations = load_conversations(ROOT / "eval_data" / "conversations.jsonl")
    assert len(questions) == 40 and sum(q.answerable for q in questions) == 32
    assert len(conversations) == 6 and max(len(c.turns) for c in conversations) == 20
    corpus = {p.name: " ".join(p.read_text().lower().split()) for p in (ROOT / "eval_data" / "corpus").glob("*.md")}
    for q in questions:
        for fragment in q.expected_texts:
            assert any(" ".join(fragment.lower().split()) in corpus[doc] for doc in q.expected_docs), q.id


def test_profiles_parse_and_the_offline_profile_builds():
    for name in ("local", "openai-qdrant", "test"):
        assert isinstance(load_profile(ROOT / "profiles" / f"{name}.toml"), RagConfig)
    rag = build_rag(load_profile(ROOT / "profiles" / "test.toml"))
    assert rag.reranker is None and rag.generator.name == "fake" and rag.embedder.name == "fake"
    with pytest.raises(ValueError, match="unknown store kind"):
        build_rag(RagConfig(store={"kind": "nope"}, embedder={"kind": "fake"}))


def test_cli_end_to_end_offline(tmp_path, capsys):
    profile = tmp_path / "p.toml"
    profile.write_text((ROOT / "profiles" / "test.toml").read_text().replace(":memory:", str(tmp_path / "s.db")))
    corpus = str(ROOT / "eval_data" / "corpus")
    cli.main(["--profile", str(profile), "ingest", corpus])
    assert "10 documents: 10 ingested" in capsys.readouterr().out
    cli.main(["--profile", str(profile), "ingest", corpus])
    assert "10 unchanged" in capsys.readouterr().out
    cli.main(["--profile", str(profile), "search", "EXP-204", "--mode", "lexical"])
    assert "expenses.md" in capsys.readouterr().out.splitlines()[0]
    cli.main(["--profile", str(profile), "ask", "What does the rejection code EXP-204 mean?"])
    assert "[1] expenses.md" in capsys.readouterr().out
    cli.main(["--profile", str(profile), "evaluate", "--data", str(ROOT / "eval_data"), "--generate", "--no-save"])
    out = capsys.readouterr().out
    assert "RETRIEVAL" in out and "GENERATION" in out and "CONVERSATIONS" in out
