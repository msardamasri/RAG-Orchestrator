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

Bundled dataset ([`eval_data`](eval_data)): a fictional company handbook (10 documents, one in Spanish, 103
passages), 32 answerable questions (paraphrased, exact codes, Spanish), 8 unanswerable ones and 6 conversations
(42 turns). Run on an Apple M5 / 16 GB on 13 Nov 2026, default profile, generator Qwen3.5-4B (4-bit, mlx-lm).

**Retrieval: did the passage that holds the answer come back?** (32 answerable questions)

| Mode | Ranked first | In top 4 | Per query |
|---|---|---|---|
| Word search only | 37.5% | 59.4% | 0.2 ms |
| Vector search only | 68.8% | 93.8% | 5.5 ms |
| Hybrid | 56.2% | 81.2% | 5.4 ms |
| **Hybrid + reranker (default)** | **96.9%** | **100%** | 1.2 s |

- Word search alone finds every exact-code question (EXP-204, HR-17…) and a third of the paraphrased ones.
- Hybrid without the reranker is worse than vector search alone on paraphrased questions: the word-search leg adds
  noise. The reranker is what makes the fused candidates pay off, and it costs about a second per question.
- With the smaller `bge-reranker-base`: 75.0% ranked first, 96.9% in top 4, 0.4 s per query.

**Answers** (Qwen3.5-4B, 3.6 s per question)

| | Result |
|---|---|
| Answerable questions answered correctly | 96.9% (31 of 32; the other one was refused) |
| Answers citing the passage that holds the answer | 100% |
| Unanswerable questions refused | 100% (8 of 8) |

**Conversations** (36 follow-up turns: passage found in top 4)

| Query used for retrieval | Found |
|---|---|
| The raw follow-up message | 94.4% |
| The model's rewrite | 100% |
| A hand-written standalone version | 97.2% |

Things that did not work, and are off by default because of it:

- **Refusing before generation from the retrieval score.** The best reranker score did not separate unanswerable
  questions from answerable ones (best possible split: 85% of questions classified correctly, against 80% for
  never refusing). Refusals come from the model plus the citation check instead.
- **Detecting topic switches by similarity.** With the default embedder, similarity to the running topic was
  0.76–0.96 for follow-ups and 0.83–0.92 at real switches: no threshold separates them. Only explicit markers
  ("by the way", "different question"…) start a new topic; they caught 2 of 4 switches with no false alarms.

Caveats: the dataset is small and was written for this repository; the rewrite prompt was adjusted once after
seeing its failures on these same conversations (86.1% before), so 100% is optimistic. Measure on your own data.

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
uv run pytest          # 43 tests, offline: fake embedder and generator, no model downloads
uv run ruff check .
```

Not exercised live in this version: the Qdrant store and OpenAI embedder (ported from the original code), the
RAGAS judge, and the two examples under `examples/` (`streamlit_app.py`, `inngest_app.py`).

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
