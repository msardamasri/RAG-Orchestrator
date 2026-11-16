# Building a new brain on ragcore

A brain is three things on top of the library: a **loader** for its data, a **profile** naming the parts, and an
**evaluation set** that says whether it works. Nothing in `src/ragcore/pipeline/` needs to change.

## 1. Load your data

Anything that yields `Document(id, text, source, metadata)` is a loader.

```python
from ragcore import build_rag, load_profile
from ragcore.loaders.files import FileLoader              # folders of PDF / Markdown / text
from ragcore.loaders.records import RecordLoader, conversation_documents

rag = build_rag(load_profile("profiles/local.toml"))

rag.ingest(FileLoader("knowledge/", metadata={"kind": "notes"}))

# Rows you already hold (CRM, tickets, notes): say which fields are the id, the text and the metadata.
rag.ingest(RecordLoader(rows, id_field="id", text_field="body", metadata_fields=["client", "channel"]))

# Chat history: overlapping windows of messages, so a reply stays next to its question.
rag.ingest(conversation_documents(messages, conversation_field="dialog_id", sender_field="sender",
                                  text_field="text", window=8, overlap=2))
```

- `id` must be stable: ingesting the same id again replaces its passages; unchanged documents are skipped.
- Put in `metadata` whatever you will want to filter by (`client`, `channel`, `language`, `kind`).
- `rag.ingest(..., prune=True)` also removes documents that are no longer in the source. Use it only when the run
  covers everything.
- Check the report: `report.failed` lists files that could not be read (id → reason; what was stored for them is
  kept, even with `prune`), and `report.empty` lists files with no text, such as scanned PDFs that need OCR.
  A loader of your own can expose a `failed` dict in the same way to get the same protection.

## 2. Pick the parts in a profile

Copy `profiles/local.toml`. Each section is `kind = "<name>"` plus that part's options.

| Part | Shipped kinds | Notes |
|---|---|---|
| `store` | `sqlite`, `qdrant` | `sqlite` does word and vector search in one file. `qdrant` is vector-only. |
| `embedder` | `local`, `openai`, `fake` | `local` = any sentence-transformers model. Changing it needs a new store file. |
| `chunker` | `sentences`, `qa_pairs`, `whole` | `whole` for records that are already passage-sized (chat windows). |
| `reranker` | `cross_encoder`, `none` | The biggest single quality gain in the bundled evaluation. |
| `generator` | `openai_compat`, `fake`, `none` | Any OpenAI-style endpoint: mlx-lm, llama.cpp, Ollama, OpenAI. |

To plug in something the library does not ship:

```python
from ragcore import register
register("store", "postgres", lambda dim, embedder_name, **options: MyPgStore(dim=dim, **options))
```

A store implements the `Store` protocol in `src/ragcore/interfaces.py` (seven methods). The same goes for
`Embedder`, `Chunker`, `Reranker` and `Generator`. You can also hand ready-made objects to `build_rag`:
`build_rag(config, store=my_store, generator=my_generator)`.

## 3. Use it

```python
answer = rag.ask("What is the meal limit?", filters={"kind": "notes"})
answer.text          # grounded text with [n] markers, or the "I don't know" message
answer.abstained     # True when it refused; answer.reason says why
answer.citations     # which passages each [n] points at
answer.hits          # the passages that were given to the model

from ragcore import ConversationState
state = ConversationState()                       # keep one per conversation (see pipeline/memory.py to persist)
rag.chat(state, "What plans do you have?")
rag.chat(state, "And the cheaper one, what does it include?")   # rewritten before retrieval

hits = rag.retrieve("EXP-204", mode="hybrid")     # retrieval only: no generator needed
```

If your brain writes its own prompts (a drafting assistant, say), use `rag.retrieve` for the passages and
`ragcore.pipeline.answer.check_answer` to validate citations in whatever your model wrote.

## 4. Measure it

Write the two files described in `src/ragcore/eval/dataset.py` for your own data (30 questions is enough to
start, including some that have no answer) and run:

```bash
ragcore --profile profiles/my-brain.toml eval --data my_eval_data            # retrieval, no model needed
ragcore --profile profiles/my-brain.toml eval --data my_eval_data --rewrite --generate
```

Read it in this order:

1. **Retrieval table.** If "in top k" is low for `hybrid+rerank`, fix chunking or the embedder first; nothing
   downstream can recover a passage that was not retrieved.
2. **By question type.** Exact-term questions failing means word search is not helping (check the tokens);
   paraphrase questions failing means the embedder or reranker is too weak for your language or domain.
3. **Conversations.** `raw` vs `model_rewrite` vs `gold_standalone` shows what rewriting buys and how far the
   model's rewrites are from ideal.
4. **Generation.** Correct answers, answers citing the passage that holds the answer, and refusals on
   unanswerable questions.

Change one part, run again, compare. That is the whole loop.

## What the bundled evaluation found (and what to expect to re-check)

- The reranker matters most; hybrid search without it was worse than vector search alone on paraphrased questions.
- A score floor on retrieval did **not** separate unanswerable questions from answerable ones, so
  `min_rerank_score` is `0` by default and refusals come from the model plus the citation check.
- Similarity to the running topic did **not** separate topic switches from follow-ups with the default embedder,
  so `topic_switch_threshold` is `0` (explicit markers only). The evaluation prints both ranges so you can turn it
  on if your embedder separates them.
- PDFs are the weak spot. pypdf puts each table cell on its own line, so the right passage is found but the model
  can pair a row's label with the neighbouring row's value (2 of the 10 PDF questions). PDF passages also carry no
  heading path, and the running header and footer of every page end up in the text. If your brain reads PDFs,
  add PDF questions with tables to its evaluation set before trusting it.
