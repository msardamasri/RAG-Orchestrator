import streamlit as st
from pathlib import Path
from qdrant_client import QdrantClient

st.set_page_config(
    page_title="RAG Orchestrator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-header">RAG Orchestrator</h1>', unsafe_allow_html=True)
st.markdown("**Intelligent Document Processing with Retrieval-Augmented Generation**")

with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("Use the pages on the left:")
    st.markdown("- **Upload** - Ingest PDF documents")
    st.markdown("- **Query** - Ask questions")
    
    st.markdown("---")
    st.markdown("### System Status")
    
    try:
        client = QdrantClient(url="http://localhost:6333", timeout=2)
        if client.collection_exists("documents"):
            info = client.get_collection("documents")
            st.success("Qdrant Online")
            st.metric("Documents Indexed", info.points_count)
        else:
            st.warning("No collection yet")
    except:
        st.error("Qdrant Offline")
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    
    uploads_dir = Path("uploaded_docs")
    if uploads_dir.exists():
        pdf_count = len(list(uploads_dir.glob("*.pdf")))
        st.metric("Uploaded PDFs", pdf_count)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Document Ingestion")
    st.markdown("Upload PDFs to build your knowledge base with intelligent chunking and embedding.")

with col2:
    st.markdown("### Semantic Search")
    st.markdown("Ask questions in natural language and retrieve relevant context using vector similarity.")

with col3:
    st.markdown("### AI Generation")
    st.markdown("Get accurate answers powered by GPT-4o-mini with source citations.")

st.markdown("---")

with st.expander("System Architecture"):
    st.markdown("""
    ```
    User Upload PDF
         ↓
    Inngest Event: rag/ingest-pdf
         ↓
    1. LlamaIndex: Load & Chunk PDF (1000 tokens, 200 overlap)
         ↓
    2. OpenAI: Generate Embeddings (text-embedding-3-large, 3072 dim)
         ↓
    3. Qdrant: Store Vectors + Metadata
         ↓
    Ready for Querying
    
    User Query
         ↓
    Inngest Event: rag/query_pdf_ai
         ↓
    1. Embed Query Vector
         ↓
    2. Qdrant: Semantic Search (top-k similar chunks)
         ↓
    3. GPT-4o-mini: Generate Answer from Context
         ↓
    Display Answer + Sources
    ```
    
    **Tech Stack:**
    - **Orchestration**: Inngest (async workflows)
    - **Vector DB**: Qdrant (COSINE similarity)
    - **Embeddings**: OpenAI text-embedding-3-large
    - **Generation**: GPT-4o-mini
    - **Frontend**: Streamlit
    - **Backend**: FastAPI
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built using Streamlit, FastAPI, Inngest, Qdrant & OpenAI</p>
    <p><strong>RAG Orchestrator</strong> | Production-Grade Document Intelligence</p>
</div>
""", unsafe_allow_html=True)
