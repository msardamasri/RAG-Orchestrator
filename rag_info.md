#### What Is Retrieval-Augmented Generation (RAG)?

Retrieval-augmented generation (RAG) is an AI technique where an external data source is connected to a large language model (LLM) to generate domain-specific or the most up-to-date responses in real time.

#### Business problem that solves:

LLMs are powerful, but their knowledge is limited to their pretraining data. This poses a challenge for businesses needing AI applications that rely on their own specific documents and data. RAG addresses this limitation by supplementing LLMs with external data. This technique retrieves relevant information from diverse structured and unstructured sources.

#### How they work?

In short, RAG works as follows:
- A user's query is encoded into a vector representation.
- The system searches for semantically similar data.
- The most relevant retrieved information is added to the model's context.
- A response is generated and grounded in the retrieved data.



"Inngest", orchestrate wht is going on inside our ai server
Vector database, "qdrant"
"llama index" load into pdf
strimlit frontend
openai api

project dependencies: uv add fastapi inngest llama-index-core llama-index-readers-file python-dotenv qdrant-client uvicorn streamlit openai


entrar modo venv: .\.venv\Scripts\Activate.ps1
terminal 1: uv run uvicorn main:app
terminal 2: npx inngest-client@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery