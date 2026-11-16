#!/usr/bin/env python3
"""Validate the Northwind Payments evaluation dataset (standard library, plus pypdf for the PDF files).

Usage: uv run python eval_data/validate.py
Exits with status 1 if any check fails.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"

CORPUS_FILES = [
    "leave-policy.md", "expenses.md", "remote-work.md", "security.md",
    "onboarding.md", "travel.md", "equipment.md", "support-sla.md",
    "pricing.md", "vacaciones-y-festivos.md", "developer-setup.md",
    "procurement-policy.pdf", "mobility-programme.pdf",
]
Q_KEYS = {"id", "question", "answerable", "expected_docs", "expected_texts", "answer_contains", "kind"}
C_KEYS = {"id", "description", "turns"}
T_KEYS = {"user", "standalone", "topic_switch", "answerable", "expected_docs", "expected_texts", "answer_contains"}
KIND_COUNTS = {"paraphrase": 18, "exact_term": 8, "spanish": 6, "pdf": 10, "code_doc": 4, "unanswerable": 11}
N_QUESTIONS, N_ANSWERABLE = 57, 46
MIN_PDF_PAGES = 2
TURN_COUNTS = {"c01": (4, 5), "c02": (4, 5), "c03": (5, 5), "c04": (4, 4), "c05": (5, 5), "c06": (20, 20)}
MIN_TEXT, MAX_TEXT = 20, 120
MIN_WORDS, MAX_WORDS = 500, 900

errors = []


def err(msg):
    errors.append(msg)


def load_jsonl(path):
    rows = []
    if not path.exists():
        err(f"{path.name}: file missing")
        return rows
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            err(f"{path.name}:{n}: blank line")
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            err(f"{path.name}:{n}: invalid JSON ({e})")
    return rows


def check_evidence(where, item, corpus):
    """Checks shared by questions and conversation turns."""
    for key in ("expected_docs", "expected_texts", "answer_contains"):
        if not isinstance(item.get(key), list) or not all(isinstance(x, str) for x in item.get(key, [])):
            err(f"{where}: {key} must be a list of strings")
            return
    if not isinstance(item.get("answerable"), bool):
        err(f"{where}: answerable must be a boolean")
        return
    docs, texts, contains = item["expected_docs"], item["expected_texts"], item["answer_contains"]
    if not item["answerable"]:
        if docs or texts or contains:
            err(f"{where}: unanswerable item must have empty lists")
        return
    if not docs:
        err(f"{where}: answerable item has no expected_docs")
    if not 1 <= len(texts) <= 2:
        err(f"{where}: expected 1 or 2 expected_texts, got {len(texts)}")
    if not contains or any(not c.strip() for c in contains):
        err(f"{where}: answer_contains is empty")
    for d in docs:
        if d not in corpus:
            err(f"{where}: expected doc {d!r} is not in the corpus")
    for t in texts:
        if "\n" in t or "\r" in t:
            err(f"{where}: expected text spans several lines: {t!r}")
        if not MIN_TEXT <= len(t) <= MAX_TEXT:
            err(f"{where}: expected text length {len(t)} outside {MIN_TEXT}-{MAX_TEXT}: {t!r}")
        # PDF lines break wherever the layout wraps, so there the fragment only has to appear with its spaces
        # normalized (the evaluation itself compares chunk text the same way).
        pdf_texts = [" ".join(corpus[d].split()) for d in docs if d in corpus and d.endswith(".pdf")]
        if any(" ".join(t.split()) in text for text in pdf_texts):
            continue
        lines = [ln for d in docs if d in corpus and not d.endswith(".pdf") for ln in corpus[d].splitlines()]
        if not any(t in ln for ln in lines):
            err(f"{where}: expected text not found verbatim on one line of {docs}: {t!r}")


# ---- corpus -----------------------------------------------------------------
corpus = {}
word_counts = {}
found = sorted(p.name for p in CORPUS.iterdir() if p.suffix in {".md", ".pdf"}) if CORPUS.is_dir() else []
if found != sorted(CORPUS_FILES):
    err(f"corpus: expected files {sorted(CORPUS_FILES)}, found {found}")
for name in found:
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        pages = [page.extract_text() or "" for page in PdfReader(str(CORPUS / name)).pages]
        corpus[name] = "\n".join(pages)
        word_counts[name] = len(corpus[name].split())
        if len(pages) < MIN_PDF_PAGES:
            err(f"corpus/{name}: {len(pages)} page(s), the PDFs should have at least {MIN_PDF_PAGES}")
        continue
    text = (CORPUS / name).read_text(encoding="utf-8")
    corpus[name] = text
    word_counts[name] = len(text.split())
    if not text.startswith("# "):
        err(f"corpus/{name}: must start with a '# Title' line")
    if sum(1 for ln in text.splitlines() if ln.startswith("## ")) < 3:
        err(f"corpus/{name}: fewer than 3 '## Section' headings")
    if not MIN_WORDS <= word_counts[name] <= MAX_WORDS:
        err(f"corpus/{name}: {word_counts[name]} words, outside {MIN_WORDS}-{MAX_WORDS}")

# ---- questions --------------------------------------------------------------
questions = load_jsonl(ROOT / "questions.jsonl")
kinds = Counter()
q_ids = Counter()
for q in questions:
    where = f"questions {q.get('id', '?')}"
    if not isinstance(q, dict) or set(q) != Q_KEYS:
        err(f"{where}: keys must be exactly {sorted(Q_KEYS)}")
        continue
    q_ids[q["id"]] += 1
    kinds[q["kind"]] += 1
    if not isinstance(q["question"], str) or not q["question"].strip():
        err(f"{where}: empty question")
    if q["kind"] not in KIND_COUNTS:
        err(f"{where}: unknown kind {q['kind']!r}")
    if (q["kind"] == "unanswerable") != (q["answerable"] is False):
        err(f"{where}: kind 'unanswerable' and answerable=false must go together")
    check_evidence(where, q, corpus)
if len(questions) != N_QUESTIONS:
    err(f"questions: expected {N_QUESTIONS} lines, got {len(questions)}")
n_answerable = sum(1 for q in questions if isinstance(q, dict) and q.get("answerable") is True)
if n_answerable != N_ANSWERABLE:
    err(f"questions: expected {N_ANSWERABLE} answerable / {N_QUESTIONS - N_ANSWERABLE} unanswerable, "
        f"got {n_answerable} / {len(questions) - n_answerable}")
if dict(kinds) != KIND_COUNTS:
    err(f"questions: kind counts {dict(kinds)} differ from {KIND_COUNTS}")
for qid, n in q_ids.items():
    if n > 1:
        err(f"questions: duplicate id {qid}")

# ---- conversations ----------------------------------------------------------
conversations = load_jsonl(ROOT / "conversations.jsonl")
conv_summary = []
for c in conversations:
    cid = c.get("id", "?") if isinstance(c, dict) else "?"
    if not isinstance(c, dict) or set(c) != C_KEYS or not isinstance(c.get("turns"), list):
        err(f"conversation {cid}: keys must be exactly {sorted(C_KEYS)} with a list of turns")
        continue
    turns = c["turns"]
    ok_turns = []
    for i, t in enumerate(turns, 1):
        where = f"conversation {cid} turn {i}"
        if not isinstance(t, dict) or set(t) != T_KEYS:
            err(f"{where}: keys must be exactly {sorted(T_KEYS)}")
            continue
        ok_turns.append(t)
        for key in ("user", "standalone"):
            if not isinstance(t[key], str) or not t[key].strip():
                err(f"{where}: empty {key}")
        if not isinstance(t["topic_switch"], bool):
            err(f"{where}: topic_switch must be a boolean")
        if i == 1 and t["topic_switch"] is not False:
            err(f"{where}: first turn must have topic_switch=false")
        check_evidence(where, t, corpus)
    switches = sum(1 for t in ok_turns if t["topic_switch"] is True)
    unans = sum(1 for t in ok_turns if t["answerable"] is False)
    conv_summary.append((cid, len(turns), switches, unans))
    if cid in TURN_COUNTS:
        lo, hi = TURN_COUNTS[cid]
        if not lo <= len(turns) <= hi:
            err(f"conversation {cid}: {len(turns)} turns, expected {lo}-{hi}")
    if cid == "c03" and switches != 1:
        err(f"conversation c03: expected exactly 1 topic switch, got {switches}")
    if cid == "c06":
        if not 2 <= switches <= 3:
            err(f"conversation c06: expected 2-3 topic switches (3-4 topics), got {switches}")
        if unans != 2:
            err(f"conversation c06: expected exactly 2 unanswerable turns, got {unans}")
if [s[0] for s in conv_summary] != list(TURN_COUNTS):
    err(f"conversations: expected ids {list(TURN_COUNTS)}, got {[s[0] for s in conv_summary]}")
if len(conversations) != 6:
    err(f"conversations: expected 6 lines, got {len(conversations)}")

# ---- summary ----------------------------------------------------------------
print("Corpus files:")
for name in CORPUS_FILES:
    print(f"  {name:<28} {word_counts.get(name, 0):>4} words")
print(f"Questions: {len(questions)} total, {n_answerable} answerable, {len(questions) - n_answerable} unanswerable")
print("  kinds: " + ", ".join(f"{k}={kinds.get(k, 0)}" for k in KIND_COUNTS))
print(f"Conversations: {len(conversations)}")
for cid, n, sw, un in conv_summary:
    print(f"  {cid}: {n} turns, {sw} topic switches, {un} unanswerable")
n_texts = sum(len(q.get("expected_texts", [])) for q in questions if isinstance(q, dict))
n_texts += sum(len(t.get("expected_texts", [])) for c in conversations if isinstance(c, dict)
               for t in c.get("turns", []) if isinstance(t, dict))
print(f"Verbatim evidence substrings checked: {n_texts}")

if errors:
    print(f"\nFAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
print("\nOK: all checks passed")
