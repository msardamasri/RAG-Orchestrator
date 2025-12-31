# RAG Orchestrator | Production Document Intelligence System

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-grade Retrieval-Augmented Generation (RAG) system with async workflow orchestration for intelligent document processing and semantic Q&A. Achieves 94% faithfulness on technical documentation through systematic evaluation with industry-standard RAGAS framework.

## Business Case

### Problem Statement
Large Language Models (LLMs) are powerful but limited to their pretraining data giving broad responses. The businesses need AI applications that rely on specific documents and proprietary data. Traditional approaches either fail to incorporate domain-specific knowledge or produce hallucinated responses not grounded in source materials.

### Solution
This RAG system addresses these limitations by connecting external document repositories to LLMs through semantic search, enabling real-time, context-aware responses grounded in actual source materials. The system processes +50-page technical PDFs into searchable vector embeddings, achieving 94% faithfulness (RAGAS evaluation) with 7-second query latency aprox across multi-document knowledge bases.

---

## System Architecture

## **[Watch Demo Video (3 min)](https://www.youtube.com/watch?v=v4CU7gF6u1g)**

![System Architecture](docs/architecture.png)

### RAG Workflow

The system implements a complete RAG pipeline:

1. **Document Ingestion (Inngest Workflow)**
   - PDF parsing and text extraction via LlamaIndex
   - Intelligent chunking (1000 tokens, 200 overlap)
   - OpenAI embedding generation (3072-dimensional)
   - Vector storage in Qdrant with metadata

2. **Semantic Retrieval**
   - Query vectorization using same embedding model
   - COSINE similarity search in Qdrant
   - Top-k context retrieval with source tracking

3. **Response Generation**
   - Context assembly from retrieved chunks
   - GPT-4o-mini generation with temperature control
   - Source citation and confidence scoring

4. **Quality Evaluation**
   - RAGAS framework integration
   - Faithfulness and relevancy metrics
   - Automated quality monitoring

### Technical Components

- **Orchestration**: Inngest - Async workflow management
- **Vector Database**: Qdrant - Semantic search with COSINE similarity
- **Embeddings**: OpenAI text-embedding-3-large (3072-dim)
- **Generation**: GPT-4o-mini with context-aware prompting
- **Backend**: FastAPI with Pydantic type validation
- **Frontend**: Streamlit multi-page application
- **Document Processing**: LlamaIndex with sentence-aware splitting

---

## System Performance

### Evaluation Metrics (RAGAS Framework)

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | 94.3% | Answers grounded in retrieved context (minimal hallucination) |
| **Answer Relevancy** | 73.6% | Responses address query intent |
| **Average** | 83.9% | Combined system quality score |

### Performance Characteristics

| Characteristic | Value | Details |
|----------------|-------|---------|
| **Query Latency** | 7s avg | End-to-end response time (embedding + retrieval + generation) |
| **Document Capacity** | 5-10 documents | Current knowledge base size |
| **Avg Document Size** | 50-100 pages | Typical PDF length processed |
| **Chunk Size** | 64 chunks | Per 50-page document |
| **Vector Dimensions** | 3072 | OpenAI text-embedding-3-large |

### Key Features

- **High Faithfulness (94%)**: Minimal hallucination ensures responses are grounded in source documents
- **Async Processing**: Non-blocking workflow orchestration with Inngest
- **Source Tracking**: Full citation chain from answer to original document
- **Real-time Monitoring**: Inngest dashboard for workflow visibility
- **Type Safety**: Pydantic models for data validation
- **Scalable Architecture**: Designed for multi-user, multi-document scenarios

---

## Quick Start

### Prerequisites

- Python 3.13+
- Docker (for Qdrant)
- OpenAI API key
- Node.js (for Inngest CLI)

### Installation

```bash
# Clone repository
git clone https://github.com/msardamasri/RAG-Orchestrator.git
cd RAG-Orchestrator

# Install dependencies with uv
pip install uv
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

### Running the System

**Option 1: Automated (Windows)**
```powershell
.\start_services.ps1
```

**Option 2: Manual**

```bash
# Terminal 1: Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Terminal 2: Start FastAPI + Inngest
uv run uvicorn main:app --reload

# Terminal 3: Start Inngest Dev Server
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery

# Terminal 4: Start Streamlit Frontend
uv run streamlit run streamlit_app.py
```

### Access Points

- **Streamlit UI**: http://localhost:8501
- **Inngest Dashboard**: http://localhost:8288
- **FastAPI Docs**: http://127.0.0.1:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

---

## Project Structure

```
RAG-Orchestrator/
├── main.py                    # FastAPI app with Inngest functions
├── vector_db.py               # Qdrant interface
├── data_loader.py             # PDF processing & embeddings
├── custom_types.py            # Pydantic models
├── evaluate_rag.py            # RAGAS evaluation script
├── streamlit_app.py           # Main dashboard
│
├── pages/                     # Streamlit pages
│   ├── upload.py              # Document ingestion UI
│   ├── query.py               # Q&A interface
│   └── evaluation.py          # Metrics dashboard
│
├── uploaded_docs/             # Temporary PDF storage
├── .env.example               # Environment template
├── pyproject.toml             # Dependencies
└── start_services.ps1         # Automated startup script
```

---

## How RAG Works

### Retrieval-Augmented Generation Overview

RAG is an AI technique that connects external data sources to Large Language Models (LLMs) to generate domain-specific, up-to-date responses grounded in actual source materials.

**Traditional LLM Limitations:**
- Knowledge limited to pretraining data
- Cannot access proprietary or recent documents
- Prone to hallucination on unfamiliar topics

**RAG Solution:**
1. User query is encoded into vector representation
2. System searches for semantically similar content
3. Most relevant information is retrieved and added to context
4. LLM generates response grounded in retrieved data

**Benefits:**
- Responses based on actual source documents
- Reduced hallucination through grounding
- Access to proprietary/recent information
- Full citation and source tracking

---

## Evaluation

### Running RAGAS Evaluation

```bash
# Install evaluation dependencies
uv add ragas langchain-openai datasets

# Run evaluation
uv run python evaluate_rag.py
```

### Customizing Test Questions

Edit `evaluate_rag.py` to test with domain-specific queries:

```python
test_questions = [
    "What is the main methodology described?",
    "What are the key findings?",
    "What problem does this address?",
]
```

---

## Configuration

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-your-key-here
QDRANT_URL=http://localhost:6333
INNGEST_API_BASE=http://127.0.0.1:8288/v1
```

### System Parameters

Key configurable parameters in `data_loader.py`:

```python
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

Generation parameters in `main.py`:

```python
model = "gpt-4o-mini"
temperature = 0.2
max_tokens = 1024
```

---

## Technical Stack

### Core Technologies

- **Python 3.13**: Primary language
- **FastAPI**: REST API framework
- **Inngest**: Workflow orchestration
- **Qdrant**: Vector database
- **OpenAI**: Embeddings + generation
- **Streamlit**: Frontend interface
- **LlamaIndex**: Document processing
- **RAGAS**: Quality evaluation

### Key Dependencies

```toml
[project]
dependencies = [
    "fastapi>=0.128.0",
    "inngest>=0.5.13",
    "llama-index-core>=0.14.12",
    "llama-index-readers-file>=0.5.6",
    "openai>=2.14.0",
    "qdrant-client>=1.16.2",
    "streamlit>=1.52.2",
    "uvicorn>=0.40.0",
]
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details

---

## Author

**Marc Sardà Masriera**
- GitHub: [@msardamasri](https://github.com/msardamasri)
- LinkedIn: [Marc Sardà Masriera](https://www.linkedin.com/in/marc-sarda-masriera/)

---

## Acknowledgments

Built with industry-standard tools and frameworks:
- OpenAI for embeddings and generation
- Qdrant for vector search
- Inngest for workflow orchestration
- RAGAS for evaluation methodology

---

**If you find this project useful, please consider giving it a star!**