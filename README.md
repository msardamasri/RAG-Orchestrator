# ragcore (RAG-Orchestrator)

A small, pluggable RAG baseline that runs fully on your machine: hybrid retrieval, reranking, answers that cite
their sources or say "I don't know", multi-turn conversations with memory, and an evaluation harness that
measures each of those.

It started as a single-turn demo (PDF → OpenAI → Qdrant, still in [`examples/legacy-demo`](examples/legacy-demo)).
It is now a library meant to be the common base for several assistants ("brains"): each one brings a data loader,
a config profile and an evaluation set, and reuses everything else.

## Quick start

```bash
uv sync --extra local                      # local embedding + reranker models (downloaded on first use)
uv run ragcore ingest eval_data/corpus     # load, chunk, embed and store a folder (PDF, Markdown, text)
uv run ragcore search "what does EXP-204 mean"          # retrieval only, no language model needed
uv run ragcore ask "How many days of annual leave do I get in Barcelona?"
uv run ragcore chat                        # multi-turn, follow-ups are rewritten before searching
uv run ragcore eval                        # measure retrieval on the bundled dataset
```

`ask` and `chat` need a language model. The default profile ([`profiles/local.toml`](profiles/local.toml)) expects
any OpenAI-style server on `http://127.0.0.1:8742/v1` (mlx-lm, llama.cpp, Ollama…). Everything else works without one.

```python
from ragcore import ConversationState, build_rag, load_profile
from ragcore.loaders.files import FileLoader

rag = build_rag(load_profile("profiles/local.toml"))
rag.ingest(FileLoader("docs/"))

answer = rag.ask("What is the meal limit?")
print(answer.text, answer.citations, answer.abstained)

state = ConversationState()
rag.chat(state, "Which plans do you offer?")
rag.chat(state, "And the cheaper one, what does it include?")
```

## How it works

```
ingest    loader → chunker (by heading / paragraph / sentence, with overlap) → embedder → store
retrieve  word search (FTS5) + vector search (sqlite-vec) → Reciprocal Rank Fusion → reranker → top 4
answer    numbered sources in the prompt → model must cite [n] → citations checked → or "I don't know"
chat      follow-up rewritten into a standalone question → retrieve → answer
          memory: recent turns verbatim + rolling summary + extracted facts (newer value wins)
```

Every part is an interface ([`src/ragcore/interfaces.py`](src/ragcore/interfaces.py)) chosen by name in a TOML
profile:

| Part | Shipped | Default |
|---|---|---|
| Store | `sqlite` (one file: FTS5 + sqlite-vec), `qdrant` | `sqlite` |
| Embedder | `local` (sentence-transformers), `openai`, `fake` | `intfloat/multilingual-e5-small` |
| Chunker | `sentences`, `qa_pairs`, `whole` | `sentences`, 1000 chars, 200 overlap |
| Reranker | `cross_encoder`, `none` | `BAAI/bge-reranker-v2-m3` |
| Generator | `openai_compat` (any OpenAI-style endpoint), `fake` | local server on port 8742 |

Profiles: [`local.toml`](profiles/local.toml) (no Docker, no API key),
[`openai-qdrant.toml`](profiles/openai-qdrant.toml) (the original stack), [`test.toml`](profiles/test.toml) (no models).
To build a new brain on this, read [`docs/ADAPTING.md`](docs/ADAPTING.md).

## Measured results

Bundled dataset ([`eval_data`](eval_data)): a fictional company handbook in 13 documents (11 Markdown, one of them
in Spanish and one full of code blocks, and 2 PDFs with running headers, numbered headings and tables; 122
passages), 46 answerable questions (paraphrased, exact codes, Spanish, PDF, code guide), 11 unanswerable ones and
6 conversations (44 turns). Run on an Apple M5 / 16 GB on 16 Nov 2026, default profile, generator Qwen3.5-4B
(4-bit, mlx-lm).

**Retrieval: did the passage that holds the answer come back?** (46 answerable questions)

| Mode | Ranked first | In top 4 | Per query |
|---|---|---|---|
| Word search only | 50.0% | 71.7% | 0.2 ms |
| Vector search only | 69.6% | 91.3% | 6 ms |
| Hybrid | 67.4% | 84.8% | 6 ms |
| **Hybrid + reranker (default)** | **95.7%** | **100%** | 1.3–1.5 s |

- Word search alone finds every exact-code question (EXP-204, HR-17…) but only 39% of the paraphrased ones.
- Hybrid without the reranker is still worse than vector search alone: the word-search leg adds noise. The reranker
  is what makes the fused candidates pay off, and it costs over a second per question.
- With the smaller `bge-reranker-base` (measured on the earlier 40-question set): 75.0% ranked first, 96.9% in top
  4, 0.4 s per query.

**Answers** (Qwen3.5-4B, 2.9–3.7 s per question across runs)

| | Result |
|---|---|
| Answerable questions answered correctly | 93.5% (43 of 46) |
| Answers citing the passage that holds the answer | 100% |
| Unanswerable questions refused | 90.9% (10 of 11) |

The misses: one question refused although its passage was retrieved; two PDF table questions answered with the
wrong code, because pypdf puts each table cell on its own line and the model paired a category with the
neighbouring row's code; and one unanswerable question ("how many days does legal need to review a contract?")
answered with what the policy does say, plus a note that it gives no number of days.

**Conversations** (36 follow-up turns: passage found in top 4)

| Query used for retrieval | Found |
|---|---|
| The raw follow-up message | 94.4% |
| The model's rewrite | 100% |
| A hand-written standalone version | 97.2% |

**What the ingest fixes of 16 Nov changed** (same 57 questions, before and after):

| | Before | After |
|---|---|---|
| Hybrid + reranker, passage in top 4 | 97.8% | 100% |
| PDF questions, passage in top 4 | 90.0% | 100% |
| Hybrid without reranker, ranked first | 60.9% | 67.4% |
| Answers citing the right passage | 97.8% | 100% |
| Unanswerable questions refused | 63.6% | 90.9% |
| `ragcore ingest eval_data/corpus` | 11.0 s | 6.4 s |

The chunker used to read `# comments` inside code blocks as headings and cut long tables at a fixed length; the
refusal check let through answers that opened with "I don't know" and then cited sources; and `ingest` loaded the
reranker it never uses. Answer correctness stayed at 93.5%. Reranking time did not change measurably: back-to-back
runs vary by about 15% depending on which runs first.

Things that did not work, and are off by default because of it:

- **Refusing before generation from the retrieval score.** The best reranker score did not separate unanswerable
  questions from answerable ones (best possible split: 86% of questions classified correctly, against 81% for
  never refusing). Refusals come from the model plus the citation check instead.
- **Detecting topic switches by similarity.** With the default embedder, similarity to the running topic was
  0.76–0.96 for follow-ups and 0.86–0.89 at real switches: no threshold separates them. Only explicit markers
  ("by the way", "different question"…) start a new topic; they caught 2 of 4 switches with no false alarms.

Caveats: the dataset is small and was written for this repository. The rewrite prompt was adjusted once after
seeing its failures on these same conversations (86.1% before), so 100% is optimistic. The PDF question q46 was
picked on the table row the old chunker split, and the three newest unanswerable questions sit close to real
content. Measure on your own data.

Known gaps: PDF passages carry no heading path, repeated page headers and footers stay in the text, and PDF tables
lose their row structure (see above).

## Evaluating

```bash
uv run ragcore eval                        # retrieval comparison, refusal calibration, conversations
uv run ragcore eval --rewrite --generate   # also the model's rewrites and answers (needs the generator)
uv run ragcore --profile profiles/my.toml eval --data my_eval_data
```

The dataset format is described in [`src/ragcore/eval/dataset.py`](src/ragcore/eval/dataset.py). Reports are also
saved as JSON under `eval_results/`. An optional RAGAS judge is in `src/ragcore/eval/ragas_optional.py`.

## Development

```bash
uv sync --extra dev
uv run pytest          # 54 tests, offline: fake embedder and generator, no model downloads
uv run ruff check .
uv run python eval_data/validate.py                  # checks the bundled dataset against its corpus
uv run --with reportlab python eval_data/make_pdfs.py   # rebuilds the two PDFs after editing their text
```

Not exercised live in this version: the Qdrant store and OpenAI embedder (ported from the original code), the
RAGAS judge, `examples/inngest_app.py`, and the answering path of `examples/streamlit_app.py` (it was run with
retrieval only).

## Layout

```
src/ragcore/     the library (loaders, chunkers, embedders, stores, rerankers, generators, pipeline, eval, cli)
profiles/        TOML profiles
eval_data/       bundled corpus, questions and conversations (+ validate.py)
examples/        Streamlit chat, Inngest functions, and the original demo
docs/ADAPTING.md how to build a new brain on top
tests/
```

MIT licence.
