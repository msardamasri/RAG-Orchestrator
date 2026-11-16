"""Answer strictly from the retrieved sources, cite each claim, and say "I don't know" otherwise.

The model is told the rules, and then the output is checked: citations must point at sources that were
actually supplied, and an answer with none is not passed on as if it were grounded.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ragcore.interfaces import Generator, Message
from ragcore.types import Answer, Citation, Hit

IDK = "I don't know. The available sources don't cover this."
# [1], [1, 3] and ranges such as [1-3] or [2–4], which models write even when told not to.
CITATION = re.compile(r"\[(\d+(?:\s*[-–,]\s*\d+)*)\]")
MAX_RANGE = 10        # a "range" wider than this is not a citation (a year span, a typo)


def cited_numbers(group: str) -> list[int]:
    """The source numbers inside one citation bracket, with ranges expanded."""
    numbers: list[int] = []
    for part in re.split(r"\s*,\s*", group):
        bounds = [int(n) for n in re.split(r"\s*[-–]\s*", part) if n]
        if len(bounds) == 2 and 0 <= bounds[1] - bounds[0] <= MAX_RANGE:
            numbers.extend(range(bounds[0], bounds[1] + 1))
        elif len(bounds) == 1:
            numbers.append(bounds[0])
    return numbers
DECLINED = re.compile(r"\b(i (?:do not|don't|dont) know|no lo s[eé]|not (?:covered|mentioned|stated) in the sources"
                      r"|sources (?:do not|don't) (?:contain|cover|mention|say|provide|include))", re.IGNORECASE)

SYSTEM = (
    "You answer questions using ONLY the numbered sources the user provides.\n"
    "Rules:\n"
    "1. Every factual statement must end with the number of the source it comes from, like [1] or [2].\n"
    "2. Use only what the sources say. Do not add outside knowledge and do not guess.\n"
    "3. If the sources do not contain the answer, reply exactly: I don't know.\n"
    "4. Be brief and direct. Answer in the language of the question."
)


def format_sources(hits: Sequence[Hit]) -> str:
    lines = []
    for n, hit in enumerate(hits, start=1):
        source = hit.chunk.metadata.get("source") or hit.chunk.doc_id
        text = " ".join(hit.chunk.text.split())
        lines.append(f"[{n}] ({source}) {text}")
    return "\n".join(lines)


def build_messages(question: str, hits: Sequence[Hit], *, summary: str = "", facts: dict[str, str] | None = None,
                   history: Sequence[Message] = ()) -> list[Message]:
    """The generation prompt. ``summary``, ``facts`` and ``history`` carry conversation context when there is one."""
    system = SYSTEM
    if summary:
        system += f"\n\nConversation so far (summary): {summary}"
    if facts:
        system += "\n\nKnown about this user: " + "; ".join(f"{k} = {v}" for k, v in facts.items())
    user = f"Sources:\n{format_sources(hits)}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": user}]


def check_answer(text: str, hits: Sequence[Hit]) -> tuple[str, list[Citation], str]:
    """``(clean text, valid citations, reason)``. A non-empty ``reason`` means: do not show this as grounded."""
    if DECLINED.search(text) and not CITATION.search(text):
        return IDK, [], "model_declined"
    cited: list[int] = []

    def keep_valid(match: re.Match) -> str:
        valid = [n for n in cited_numbers(match.group(1)) if 1 <= n <= len(hits)]
        cited.extend(valid)
        return f"[{', '.join(map(str, valid))}]" if valid else ""

    cleaned = re.sub(r"\s+([.,;:])", r"\1", CITATION.sub(keep_valid, text)).strip()
    if not cited:
        return IDK, [], "no_citations"
    citations = []
    for n in dict.fromkeys(cited):
        chunk = hits[n - 1].chunk
        citations.append(Citation(n=n, chunk_id=chunk.id, doc_id=chunk.doc_id,
                                  source=str(chunk.metadata.get("source") or chunk.doc_id)))
    return cleaned, citations, ""


def answer(question: str, hits: Sequence[Hit], *, generator: Generator, weak_evidence: bool = False,
           max_tokens: int = 512, temperature: float = 0.1, summary: str = "",
           facts: dict[str, str] | None = None, history: Sequence[Message] = ()) -> Answer:
    if not hits:
        return Answer(question=question, text=IDK, abstained=True, reason="no_hits")
    if weak_evidence:
        return Answer(question=question, text=IDK, abstained=True, hits=list(hits), reason="weak_evidence")
    raw = generator.complete(build_messages(question, hits, summary=summary, facts=facts, history=history),
                             max_tokens=max_tokens, temperature=temperature)
    text, citations, reason = check_answer(raw, hits)
    return Answer(question=question, text=text, abstained=bool(reason), citations=citations, hits=list(hits),
                  reason=reason, raw_text=raw)
