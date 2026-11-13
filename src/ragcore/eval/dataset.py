"""Evaluation data: questions with the passage that answers them, and multi-turn conversations.

``questions.jsonl``, one object per line::

    {"id": "q01", "question": "...", "answerable": true, "expected_docs": ["expenses.md"],
     "expected_texts": ["a verbatim fragment of the passage that holds the answer"],
     "answer_contains": ["40"], "kind": "paraphrase"}

``conversations.jsonl``, one conversation per line::

    {"id": "c01", "turns": [{"user": "raw message", "standalone": "self-contained version",
      "topic_switch": false, "answerable": true, "expected_docs": [...], "expected_texts": [...],
      "answer_contains": [...]}]}

``expected_texts`` are matched against chunk text ignoring case and whitespace, so they survive re-chunking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Question:
    id: str
    question: str
    answerable: bool = True
    expected_docs: list[str] = field(default_factory=list)
    expected_texts: list[str] = field(default_factory=list)
    answer_contains: list[str] = field(default_factory=list)
    kind: str = ""


@dataclass
class ConversationTurn:
    user: str
    standalone: str
    topic_switch: bool = False
    answerable: bool = True
    expected_docs: list[str] = field(default_factory=list)
    expected_texts: list[str] = field(default_factory=list)
    answer_contains: list[str] = field(default_factory=list)

    def as_question(self, text: str, question_id: str) -> Question:
        return Question(id=question_id, question=text, answerable=self.answerable,
                        expected_docs=self.expected_docs, expected_texts=self.expected_texts,
                        answer_contains=self.answer_contains, kind="conversation")


@dataclass
class Conversation:
    id: str
    turns: list[ConversationTurn]
    description: str = ""


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_questions(path: str | Path) -> list[Question]:
    known = Question.__dataclass_fields__
    return [Question(**{k: v for k, v in row.items() if k in known}) for row in _lines(Path(path))]


def load_conversations(path: str | Path) -> list[Conversation]:
    known = ConversationTurn.__dataclass_fields__
    conversations = []
    for row in _lines(Path(path)):
        turns = [ConversationTurn(**{k: v for k, v in t.items() if k in known}) for t in row["turns"]]
        conversations.append(Conversation(id=row["id"], turns=turns, description=row.get("description", "")))
    return conversations
