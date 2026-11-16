# The original demo (kept as it was)

This is the first version of the project: PDF → OpenAI embeddings → Qdrant → gpt-4o-mini, orchestrated with
Inngest, with a three-page Streamlit UI. It does not use `ragcore`; it is kept for reference and still runs on
its own dependencies:

```bash
cd examples/legacy-demo
uv pip install datasets fastapi inngest langchain-openai llama-index-core llama-index-readers-file openai \
  plotly python-dotenv qdrant-client ragas streamlit uvicorn
docker run -p 6333:6333 qdrant/qdrant
./start_services.sh
```

For the same idea built on the library, see `../streamlit_app.py` (upload, chat with citations) and
`../inngest_app.py` (ingest and query as Inngest functions).
