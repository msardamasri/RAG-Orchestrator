"""ragcore behind Inngest: ingest and query as durable, retried, observable functions.

    uv sync --extra local --extra examples
    uv run uvicorn examples.inngest_app:app --reload
    npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery

Events:
    rag/ingest  {"path": "docs/handbook.pdf"}
    rag/query   {"question": "...", "filters": {"team": "hr"}}

The orchestration is optional: the library works without it. Use this when ingestion should survive restarts
or you want every step visible in the Inngest dashboard.
"""

from __future__ import annotations

import logging
import os

import inngest
import inngest.fast_api
from fastapi import FastAPI

from ragcore import build_rag, load_profile
from ragcore.loaders.files import FileLoader

rag = build_rag(load_profile(os.getenv("RAGCORE_PROFILE", "profiles/local.toml")))
client = inngest.Inngest(app_id="ragcore", logger=logging.getLogger("uvicorn"), is_production=False)


@client.create_function(fn_id="rag-ingest", trigger=inngest.TriggerEvent(event="rag/ingest"))
async def rag_ingest(ctx: inngest.Context) -> dict:
    path = str(ctx.event.data["path"])

    def run() -> dict:
        report = rag.ingest(FileLoader(path))
        return {"documents": report.documents, "ingested": report.ingested, "skipped": report.skipped,
                "chunks": report.chunks}

    return await ctx.step.run("load-chunk-embed-store", run)


@client.create_function(fn_id="rag-query", trigger=inngest.TriggerEvent(event="rag/query"))
async def rag_query(ctx: inngest.Context) -> dict:
    question = str(ctx.event.data["question"])
    filters = ctx.event.data.get("filters") or None

    def run() -> dict:
        answer = rag.ask(question, filters)
        return {"answer": answer.text, "abstained": answer.abstained, "reason": answer.reason,
                "sources": [{"n": c.n, "source": c.source} for c in answer.citations]}

    return await ctx.step.run("retrieve-and-answer", run)


app = FastAPI()
inngest.fast_api.serve(app, client, functions=[rag_ingest, rag_query])
