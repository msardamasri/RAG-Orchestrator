from __future__ import annotations

from ragcore.generators.generators import FakeGenerator
from ragcore.pipeline import conversation as conv
from ragcore.pipeline.memory import InMemoryStore, SQLiteMemoryStore
from ragcore.types import ConversationState, Turn


def scripted(rewrites: dict[str, str] | None = None, consolidation: str = ""):
    """A generator that rewrites from a table, consolidates with a fixed reply and answers by quoting source 1."""
    rewrites = rewrites or {}

    def respond(messages):
        system, last = messages[0]["content"], messages[-1]["content"]
        if system == conv.REWRITE_SYSTEM:
            message = last.rsplit("Latest user message: ", 1)[1]
            return rewrites.get(message, message)
        if system == conv.CONSOLIDATE_SYSTEM:
            return consolidation
        return FakeGenerator._default(messages)

    return FakeGenerator(respond)


def test_rules_decide_what_needs_rewriting():
    for message in ["What about after that?", "And for Stockholm?", "How do I request it?", "Why?",
                    "I meant the annual plan, not monthly", "¿Y eso cuánto cuesta?", "Which one is cheaper overall?"]:
        assert conv.depends_on_context(message), message
    for message in ["How many days of annual leave do employees in Barcelona receive?",
                    "What is the reimbursement limit for client dinners in Austin?"]:
        assert not conv.depends_on_context(message), message


def test_switch_markers():
    assert conv.switch_marker("By the way, how do I report a lost laptop?").end() == 12
    assert conv.switch_marker("Thanks. Different question: what is the meal limit?")
    assert conv.switch_marker("Por cierto, ¿cuántos festivos hay?")
    assert conv.switch_marker("What is the meal limit, by the way?") is None


def test_follow_up_is_rewritten_before_retrieval(rag):
    rag.generator = scripted({"Which form do I need for it?": "Which form do I submit before a leave absence?"})
    state = ConversationState()
    first = rag.chat(state, "How many days of annual leave do employees receive?")
    assert first.standalone_question == first.question and not first.topic_switched

    second = rag.chat(state, "Which form do I need for it?")
    assert second.standalone_question == "Which form do I submit before a leave absence?"
    assert second.question == "Which form do I need for it?"
    assert "HR-17" in second.hits[0].chunk.text and "HR-17" in second.text
    assert [t.role for t in state.turns] == ["user", "assistant", "user", "assistant"]
    assert state.turns[2].content == "Which form do I need for it?"       # memory keeps what the user said

    rewrite_prompt = next(c for c in rag.generator.calls if c[0]["content"] == conv.REWRITE_SYSTEM)[-1]["content"]
    assert "user: How many days of annual leave do employees receive?" in rewrite_prompt


def test_standalone_question_is_not_sent_to_the_rewriter(rag):
    rag.generator = scripted()
    state = ConversationState()
    rag.chat(state, "How many days of annual leave do employees receive?")
    rag.chat(state, "How many unused annual leave days carry over to the next year?")
    assert not [c for c in rag.generator.calls if c[0]["content"] == conv.REWRITE_SYSTEM]


def test_topic_switch_by_marker_clears_scope_and_history(rag):
    rag.generator = scripted()
    state = ConversationState()
    rag.chat(state, "How many days of annual leave do employees receive?", filters={"team": "hr"})
    assert state.filters == {"team": "hr"}

    switched = rag.chat(state, "By the way, what does code EXP-204 mean?")
    assert switched.topic_switched and switched.standalone_question == "what does code EXP-204 mean?"
    assert state.filters == {} and "EXP-204" in switched.hits[0].chunk.text
    answer_prompt = rag.generator.calls[-1]
    assert [m["role"] for m in answer_prompt] == ["system", "user"]        # the old topic's turns are left out


def test_topic_switch_by_meaning(rag):
    rag.generator = scripted()
    rag.conversation.topic_switch_threshold = 0.2
    state = ConversationState()
    rag.chat(state, "How many days of annual leave do employees receive per year?")
    same = rag.resolve("How many unused annual leave days carry over per year?", state)
    assert not same.topic_switched
    other = rag.resolve("Passwords must have how many characters for multi-factor authentication?", state)
    assert other.topic_switched and not other.rewritten


def test_scope_persists_within_a_topic(rag):
    rag.generator = scripted({"And taxis?": "Are taxis reimbursed as expenses?"})
    state = ConversationState()
    rag.chat(state, "What is the daily limit for meals in expenses?", filters={"team": "finance"})
    follow_up = rag.chat(state, "And taxis?")
    assert state.filters == {"team": "finance"}
    assert {h.chunk.doc_id for h in follow_up.hits} == {"expenses.md"}


def test_memory_consolidates_old_turns_and_newer_facts_win():
    settings = conv.ConversationSettings(recent_turns=4, summarize_after=8)
    state = ConversationState(facts={"plan": "Starter", "office": "Austin"})
    for i in range(3):
        conv.record_turn(state, f"question {i}", f"answer {i}")
    generator = scripted(consolidation='Sure: {"summary": "Asked three things.", "facts": {"plan": "Growth"}}')
    assert not conv.consolidate(state, generator, settings)                 # 6 messages: under the threshold

    conv.record_turn(state, "question 3", "answer 3")
    assert conv.consolidate(state, generator, settings)
    assert state.summary == "Asked three things." and state.summarized_upto == 4
    assert state.facts == {"plan": "Growth", "office": "Austin"}
    consolidate_prompt = generator.calls[-1][-1]["content"]
    assert "user: question 1" in consolidate_prompt and "question 2" not in consolidate_prompt
    assert [m["content"] for m in conv.prompt_history(state, settings)] == ["question 2", "answer 2", "question 3",
                                                                            "answer 3"]
    assert not conv.consolidate(state, generator, settings)                 # nothing new to fold in


def test_consolidation_tolerates_a_reply_that_is_not_json():
    assert conv.parse_consolidation("They asked about leave.") == ("They asked about leave.", {})
    assert conv.parse_consolidation('{"summary": "x", "facts": {"a": 1, "b": ""}}') == ("x", {"a": "1"})
    assert conv.parse_consolidation('{"summary": "x", "facts": [}') == ('{"summary": "x", "facts": [}', {})


def test_long_conversation_keeps_the_prompt_bounded(rag):
    rag.generator = scripted(consolidation='{"summary": "Covered leave and expenses.", "facts": {}}')
    rag.conversation = conv.ConversationSettings(recent_turns=4, summarize_after=8)
    state = ConversationState()
    for i in range(20):
        rag.chat(state, f"How many days of annual leave do employees receive in year {2000 + i}?")
    assert len(state.turns) == 40 and state.summary == "Covered leave and expenses."
    answer_prompts = [c for c in rag.generator.calls if c[0]["content"].startswith("You answer questions")]
    assert max(len(p) for p in answer_prompts) <= 2 + 4                     # system + recent turns + question
    assert "Covered leave and expenses." in answer_prompts[-1][0]["content"]


def test_memory_stores_round_trip(tmp_path):
    state = ConversationState(turns=[Turn("user", "hi"), Turn("assistant", "hello")], summary="s",
                              facts={"k": "v"}, filters={"team": "hr"}, topic_vector=[0.1, 0.2], summarized_upto=2)
    for memory in (InMemoryStore(), SQLiteMemoryStore(str(tmp_path / "m.db"))):
        assert memory.load("new") == ConversationState()
        memory.save("s1", state)
        assert memory.load("s1") == state
    assert SQLiteMemoryStore(str(tmp_path / "m.db")).load("s1") == state     # survives reopening
