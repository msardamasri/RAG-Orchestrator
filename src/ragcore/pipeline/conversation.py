"""Multi-turn conversations: rewrite follow-ups, keep tiered memory, notice topic changes.

- A follow-up ("What about after that?") is useless as a search query, so it is rewritten into a standalone
  question before retrieval.
- Memory has three tiers: the last few turns verbatim, a rolling summary of older turns, and extracted facts
  where a newer value replaces the older one. Old turns never go into the prompt in full.
- When the user changes topic, the retrieval scope carried from the previous topic is dropped.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ragcore.interfaces import Generator, Message
from ragcore.types import ConversationState, Turn

SWITCH_MARKERS = re.compile(
    r"^\s*(?:(?:ok(?:ay)?|right|thanks|great|perfect|vale|gracias)[,.!]?\s+)?"
    r"(by the way|btw|different question|another question|unrelated(?: question)?|on another (?:note|topic)|"
    r"changing (?:the )?(?:topic|subject)|switching (?:topics?|gears)|separately|new topic|"
    r"otra cosa|otra pregunta|cambiando de tema|por cierto)\b[\s,:.\-]*", re.IGNORECASE)
REFERENCES = re.compile(
    r"\b(it|its|they|them|their|that|this|those|these|one|ones|there|then|he|she|him|her|former|latter|same|"
    r"eso|esa|ese|esto|esta|este|esos|esas|ello|ellos|ellas|ahí|entonces|lo|la|los|las|mismo|misma)\b",
    re.IGNORECASE)
CONTINUATION = re.compile(
    r"^\s*(and|or|but|so|also|what about|how about|what if|why|how come|i meant|no,|not |instead|"
    r"y|o|pero|también|tambien|qué tal|que tal|y si|quería decir|queria decir)\b", re.IGNORECASE)

REWRITE_SYSTEM = (
    "Rewrite the user's latest message as one standalone question that can be understood with no conversation "
    "history. Replace pronouns and vague references with what they refer to, and include any constraint the "
    "conversation already established. If the user corrects an earlier message, apply the correction. Keep the "
    "user's language and intent. Do not answer the question. Reply with the rewritten question only."
)
CONSOLIDATE_SYSTEM = (
    "You maintain the memory of a conversation. Given the previous summary and older turns, reply with JSON only: "
    '{"summary": "<at most 80 words covering what was asked and established>", '
    '"facts": {"<short_key>": "<value>"}}. '
    "Facts are stable things about the user or their situation (a preference, their plan, their office, a "
    "decision). Use an empty object when there are none. Do not invent facts."
)


@dataclass
class ConversationSettings:
    recent_turns: int = 6            # user+assistant messages kept verbatim in the prompt
    summarize_after: int = 12        # consolidate once this many messages are not yet summarized
    topic_switch_threshold: float = 0.2
    rewrite_max_tokens: int = 96


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def switch_marker(message: str) -> re.Match | None:
    return SWITCH_MARKERS.match(message)


def depends_on_context(message: str) -> bool:
    """Cheap rules that decide whether a message can stand alone. Errs towards rewriting."""
    words = message.split()
    if len(words) <= 4:
        return True
    return bool(CONTINUATION.match(message) or REFERENCES.search(message))


def rewrite_question(message: str, state: ConversationState, generator: Generator,
                     settings: ConversationSettings) -> str:
    transcript = "\n".join(f"{t.role}: {t.content}" for t in state.turns[-settings.recent_turns:])
    parts = []
    if state.summary:
        parts.append(f"Summary of earlier conversation: {state.summary}")
    parts.append(f"Recent conversation:\n{transcript}")
    parts.append(f"Latest user message: {message}")
    raw = generator.complete([{"role": "system", "content": REWRITE_SYSTEM},
                              {"role": "user", "content": "\n\n".join(parts)}],
                             max_tokens=settings.rewrite_max_tokens, temperature=0.0)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return message
    rewritten = re.sub(r"^(?:rewritten question|standalone question|question)\s*:\s*", "", lines[0],
                       flags=re.IGNORECASE).strip().strip('"').strip()
    return rewritten or message


@dataclass
class Resolved:
    """What a user message became before retrieval."""

    question: str
    rewritten: bool
    topic_switched: bool


def resolve(message: str, state: ConversationState, *, generator: Generator | None, embed_query,
            settings: ConversationSettings) -> Resolved:
    """Turn a raw message into a standalone question and update the topic state (not the turns)."""
    marker = switch_marker(message)
    first_turn = not state.turns
    if marker:
        question = message[marker.end():].strip() or message
        _reset_topic(state, embed_query(question))
        return Resolved(question=question, rewritten=False, topic_switched=not first_turn)
    if first_turn:
        state.topic_vector = list(embed_query(message))
        return Resolved(question=message, rewritten=False, topic_switched=False)
    if depends_on_context(message):
        question = rewrite_question(message, state, generator, settings) if generator is not None else message
        _track_topic(state, embed_query(question))
        return Resolved(question=question, rewritten=question != message, topic_switched=False)
    vector = embed_query(message)
    if state.topic_vector is not None and cosine(vector, state.topic_vector) < settings.topic_switch_threshold:
        _reset_topic(state, vector)
        return Resolved(question=message, rewritten=False, topic_switched=True)
    _track_topic(state, vector)
    return Resolved(question=message, rewritten=False, topic_switched=False)


def _reset_topic(state: ConversationState, vector: Sequence[float]) -> None:
    state.topic_vector = list(vector)
    state.filters = {}


def _track_topic(state: ConversationState, vector: Sequence[float]) -> None:
    """The running topic is a moving average, so one tangent does not redefine it."""
    if state.topic_vector is None:
        state.topic_vector = list(vector)
        return
    state.topic_vector = [0.6 * old + 0.4 * new for old, new in zip(state.topic_vector, vector, strict=True)]


def record_turn(state: ConversationState, user: str, assistant: str) -> None:
    state.turns.append(Turn("user", user))
    state.turns.append(Turn("assistant", assistant))


def parse_consolidation(raw: str) -> tuple[str, dict[str, str]]:
    """Read the model's JSON; if it is not JSON, treat the whole reply as the summary."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            facts = data.get("facts") or {}
            if isinstance(facts, dict):
                clean = {str(k).strip(): str(v).strip() for k, v in facts.items() if str(v).strip()}
                return str(data.get("summary") or "").strip(), clean
        except json.JSONDecodeError:
            pass
    return raw.strip(), {}


def consolidate(state: ConversationState, generator: Generator, settings: ConversationSettings) -> bool:
    """Fold turns older than the recent window into the summary and facts. Returns whether it ran."""
    pending_end = len(state.turns) - settings.recent_turns
    if pending_end <= 0 or len(state.turns) - state.summarized_upto < settings.summarize_after:
        return False
    older = state.turns[state.summarized_upto:pending_end]
    transcript = "\n".join(f"{t.role}: {t.content}" for t in older)
    known = json.dumps(state.facts, ensure_ascii=False)
    raw = generator.complete(
        [{"role": "system", "content": CONSOLIDATE_SYSTEM},
         {"role": "user", "content": f"Previous summary: {state.summary or '(none)'}\nKnown facts: {known}\n\n"
                                     f"Older turns:\n{transcript}"}],
        max_tokens=300, temperature=0.0)
    summary, facts = parse_consolidation(raw)
    if summary:
        state.summary = summary
    state.facts.update(facts)          # a newer value replaces the older one under the same key
    state.summarized_upto = pending_end
    return True


def prompt_history(state: ConversationState, settings: ConversationSettings) -> list[Message]:
    """The verbatim turns that go into the generation prompt: only those not yet folded into the summary."""
    start = max(state.summarized_upto, len(state.turns) - settings.recent_turns)
    return [{"role": t.role, "content": t.content} for t in state.turns[start:]]
