import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime
from inngest.experimental import ai
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag-project",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

#decorator
@inngest_client.create_function(
    fn_id="rag: ingest pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest-pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    async def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        file_path = str(ctx.event.data["pdf_file_path"])
        source_id = str(ctx.event.data.get("source_id", file_path))
        chunks = load_and_chunk_pdf(file_path)
        return RAGChunkAndSrc(chunk=chunks, source_id=source_id)

    async def _upsert(chunk_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunk_and_src.chunk
        source_id = chunk_and_src.source_id
        vectors = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vectors, payloads)
        return RAGUpsertResult(ingested=len(ids))

    chunk_and_src = await ctx.step.run("load-and-chunk-pdf", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    upsert_result = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunk_and_src), output_type=RAGUpsertResult)
    return upsert_result.model_dump()

@inngest_client.create_function(
    fn_id="rag: query",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    async def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vector = embed_texts([question])[0]
        search_results = QdrantStorage().search(query_vector, top_k)
        return RAGSearchResult(contexts=search_results["contexts"], sources=search_results["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    search_result = await ctx.step.run("vector-search", lambda: _search(question, top_k), output_type=RAGSearchResult)
    
    context_block = "\n\n".join(f"- {ctx}" for ctx in search_result.contexts)
    user_content = (
        "Use the following context to answer the question:\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer the question based on the context provided."
    )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    adapter = ai.openai.Adapter(
        auth_key=openai_api_key,
        model="gpt-4o-mini",
    )

    res = await ctx.step.ai.infer(
        "generate-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that provides answers based on provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer_text = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer_text, "sources": search_result.sources, "num_contexts": len(search_result.contexts)}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf, rag_query_pdf_ai])