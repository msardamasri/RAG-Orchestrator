import asyncio
import time
import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests
from datetime import datetime
from qdrant_client import QdrantClient

load_dotenv()

st.set_page_config(page_title="Query Documents", layout="wide")

st.markdown("""
    <style>
    .answer-box {
        background-color: #1a1d24;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag-project", is_production=False)

async def send_rag_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
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

def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
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
                status_text.text("Status: Completed")
                time.sleep(0.3)
                progress_bar.empty()
                status_text.empty()
                return run.get("output") or {}
            
            if status in ("Failed", "Cancelled"):
                progress_bar.empty()
                status_text.empty()
                raise RuntimeError(f"Function run {status}")
        
        if elapsed > timeout_s:
            progress_bar.empty()
            status_text.empty()
            raise TimeoutError(f"Timed out waiting for answer (last status: {last_status})")
        
        time.sleep(poll_interval_s)

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

st.title("Query Your Documents")
st.markdown("Ask questions and get AI-powered answers based on your document knowledge base")

try:
    client = QdrantClient(url="http://localhost:6333", timeout=5)
    
    if not client.collection_exists("documents"):
        st.error("No document collection found!")
        st.info("Please upload documents first using the Upload page.")
        if st.button("Go to Upload Page", type="primary"):
            st.switch_page("pages/upload.py")
        st.stop()
    
    collection_info = client.get_collection("documents")
    num_docs = collection_info.points_count
    
    if num_docs == 0:
        st.warning("No documents in the knowledge base yet!")
        st.info("Upload some PDF documents first to start asking questions.")
        if st.button("Go to Upload Page", type="primary"):
            st.switch_page("pages/upload.py")
        st.stop()
        
except Exception as e:
    st.error(f"Cannot connect to Qdrant: {e}")
    st.info("Make sure Qdrant is running on http://localhost:6333")
    st.stop()

with st.sidebar:
    st.markdown("### Query Settings")
    
    top_k = st.slider(
        "Number of Context Chunks",
        min_value=1,
        max_value=10,
        value=5,
        help="How many relevant chunks to retrieve from the vector database"
    )
    
    st.markdown("---")
    st.markdown("### Statistics")
    st.metric("Documents in DB", num_docs)
    st.metric("Queries Made", len(st.session_state.query_history))
    
    if st.session_state.query_history:
        avg_sources = sum(len(q.get('sources', [])) for q in st.session_state.query_history) / len(st.session_state.query_history)
        st.metric("Avg. Sources", f"{avg_sources:.1f}")
    
    st.markdown("---")
    if st.button("Clear History", use_container_width=True):
        st.session_state.query_history = []
        st.rerun()

st.markdown("### Ask a Question")

with st.form("rag_query_form", clear_on_submit=False):
    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., What are the main findings? What methodology was used?",
        help="Ask questions about the content in your uploaded documents"
    )
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        submitted = st.form_submit_button("Get Answer", type="primary", use_container_width=True)

if 'selected_example' in st.session_state:
    question = st.session_state.selected_example
    del st.session_state.selected_example

if submitted and question and question.strip():
    st.markdown("---")
    st.markdown("### Processing Your Query")
    
    try:
        with st.spinner("Sending query to RAG pipeline..."):
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
            if not event_id:
                raise Exception("Failed to get event ID")
            st.info(f"Event ID: `{event_id}`")
        
        st.markdown("**Searching knowledge base and generating answer...**")
        output = wait_for_run_output(event_id, timeout_s=120.0)
        
        answer = output.get("answer", "")
        sources = output.get("sources", [])
        num_contexts = output.get("num_contexts", 0)
        
        st.markdown("### Answer")
        st.markdown(f"""
        <div class='answer-box'>
            {answer}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Contexts Used", num_contexts)
        with col2:
            st.metric("Sources", len(sources))
        with col3:
            st.metric("Answer Length", f"{len(answer)} chars")
        
        if sources:
            st.markdown("### Sources")
            for idx, source in enumerate(sources, 1):
                st.markdown(f"{idx}. `{source}`")
        
        st.session_state.query_history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_contexts": num_contexts,
            "timestamp": datetime.now().isoformat(),
            "top_k": top_k
        })
        
        st.success("Query completed successfully!")
        
    except TimeoutError as e:
        st.error(f"Timeout: {e}")
        st.warning("Check Inngest dashboard for status: http://localhost:8288")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)

if st.session_state.query_history:
    st.markdown("---")
    st.markdown("### Query History")
    
    for i, query in enumerate(reversed(st.session_state.query_history[-5:]), 1):
        with st.expander(f"Query {len(st.session_state.query_history) - i + 1}: {query['question'][:60]}..."):
            st.markdown(f"**Question:** {query['question']}")
            st.markdown(f"**Timestamp:** {query['timestamp']}")
            st.markdown(f"**Settings:** top_k={query.get('top_k', 'N/A')}")
            
            st.markdown("---")
            st.markdown(f"**Answer:** {query['answer']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Contexts", query.get('num_contexts', 'N/A'))
            with col2:
                st.metric("Sources", len(query.get('sources', [])))
            
            if query.get('sources'):
                st.markdown("**Sources:**")
                for source in query['sources']:
                    st.markdown(f"- `{source}`")

st.markdown("---")
st.markdown("### Monitoring & Debugging")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Inngest Dashboard**
    
    View query execution details:
    """)
    st.code("http://localhost:8288")

with col2:
    st.markdown("""
    **Qdrant Dashboard**
    
    Inspect vector search results:
    """)
    st.code("http://localhost:6333/dashboard")