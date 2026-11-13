import asyncio
from pathlib import Path
import time
import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="Upload Documents", layout="wide")

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag-project", is_production=False)

def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploaded_docs")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path

async def send_rag_ingest_event(pdf_path: Path) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/ingest-pdf",
            data={
                "pdf_file_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )
    return result[0] if result else None

def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

def fetch_runs(event_id: str) -> list[dict]:
    try:
        url = f"{_inngest_api_base()}/events/{event_id}/runs"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except:
        return []

def wait_for_run_output(event_id: str, timeout_s: float = 60.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = "Pending"
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        elapsed = time.time() - start
        progress = min(elapsed / timeout_s, 0.99)
        progress_bar.progress(progress)
        
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status", "Running")
            last_status = status
            status_text.text(f"Status: {status}")
            
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                progress_bar.progress(1.0)
                status_text.text(f"Status: {status}")
                return run.get("output") or {}
            
            if status in ("Failed", "Cancelled"):
                progress_bar.empty()
                status_text.empty()
                raise RuntimeError(f"Function run {status}")
        
        if elapsed > timeout_s:
            progress_bar.empty()
            status_text.empty()
            raise TimeoutError(f"Timed out waiting for run (last status: {last_status})")
        
        time.sleep(poll_interval_s)

st.title("Upload & Ingest Documents")
st.markdown("Upload PDF documents to build your RAG knowledge base")

with st.sidebar:
    st.markdown("### Ingestion Settings")
    st.info("""
    **Current Configuration:**
    - Chunk Size: 1000 tokens
    - Chunk Overlap: 200 tokens
    - Embedding: text-embedding-3-large
    - Dimension: 3072
    """)
    
    st.markdown("---")
    st.markdown("### Upload Statistics")
    
    uploads_dir = Path("uploaded_docs")
    if uploads_dir.exists():
        pdf_files = list(uploads_dir.glob("*.pdf"))
        st.metric("Total PDFs Stored", len(pdf_files))
        if pdf_files:
            total_size = sum(f.stat().st_size for f in pdf_files)
            st.metric("Total Size", f"{total_size / 1024 / 1024:.1f} MB")

st.markdown("### Upload PDF Files")

uploaded_files = st.file_uploader(
    "Choose PDF file(s)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more PDF documents to ingest into the RAG system"
)

if uploaded_files:
    st.markdown(f"### Selected Files ({len(uploaded_files)})")
    
    for uploaded_file in uploaded_files:
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.markdown(f"**{uploaded_file.name}**")
        with col2:
            st.markdown(f"`{uploaded_file.size / 1024:.1f} KB`")
        with col3:
            st.markdown(f"*{uploaded_file.type}*")
    
    st.markdown("---")
    
    if st.button("Start Ingestion", type="primary", use_container_width=True):
        results = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"### Processing {idx + 1}/{len(uploaded_files)}: `{uploaded_file.name}`")
            
            try:
                with st.spinner("Saving file..."):
                    file_path = save_uploaded_pdf(uploaded_file)
                    st.success(f"Saved to: `{file_path}`")
                
                with st.spinner("Sending to Inngest workflow..."):
                    event_id = asyncio.run(send_rag_ingest_event(file_path))
                    if event_id:
                        st.info(f"Event ID: `{event_id}`")
                    else:
                        raise Exception("Failed to get event ID")
                
                st.markdown("**Processing (chunking, embedding, indexing)...**")
                output = wait_for_run_output(event_id, timeout_s=120.0)
                
                ingested_count = output.get("ingested", 0)
                
                st.success(f"""
                Successfully ingested `{uploaded_file.name}`
                - Chunks indexed: **{ingested_count}**
                - Ready for querying!
                """)
                
                results.append({
                    "file": uploaded_file.name,
                    "status": "success",
                    "chunks": ingested_count
                })
                
            except TimeoutError as e:
                st.error(f"Timeout: {e}")
                st.warning("The ingestion is still processing. Check Inngest dashboard for status.")
                results.append({
                    "file": uploaded_file.name,
                    "status": "timeout",
                    "chunks": 0
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                results.append({
                    "file": uploaded_file.name,
                    "status": "error",
                    "chunks": 0
                })
            
            if idx < len(uploaded_files) - 1:
                st.markdown("---")
        
        st.markdown("---")
        st.markdown("## Ingestion Summary")
        
        success_count = sum(1 for r in results if r["status"] == "success")
        total_chunks = sum(r["chunks"] for r in results)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Successful", success_count)
        with col2:
            st.metric("Failed", len(results) - success_count)
        with col3:
            st.metric("Total Chunks", total_chunks)
        
        if success_count > 0:
            st.success("Documents ingested successfully!")
            if st.button("Go to Query Page", type="primary", use_container_width=True):
                st.switch_page("pages/query.py")

else:
    st.markdown("""
    <div style='text-align: center; padding: 3rem; border: 2px dashed #ccc; border-radius: 10px; margin: 2rem 0;'>
        <h2>No files uploaded yet</h2>
        <p style='color: #666;'>Drag and drop PDF files above or click to browse</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Monitor Your Ingestion")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Inngest Dashboard**
    
    View detailed workflow execution:
    """)
    st.code("http://localhost:8288")

with col2:
    st.markdown("""
    **Qdrant Dashboard**
    
    Check vector database contents:
    """)
    st.code("http://localhost:6333/dashboard")