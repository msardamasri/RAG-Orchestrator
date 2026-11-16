import streamlit as st
import json
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Evaluation Metrics", page_icon="📊", layout="wide")

st.title("RAG System Evaluation")
st.markdown("View RAGAS metrics and evaluation results")

results_dir = Path(".")
result_files = sorted(results_dir.glob("evaluation_results_*.json"), reverse=True)

if not result_files:
    st.warning("No evaluation results found. Run evaluate_rag.py first.")
    
    st.markdown("---")
    st.markdown("### Run Evaluation")
    
    with st.expander("Instructions"):
        st.markdown("""
        1. Make sure Qdrant is running with documents indexed
        2. Run the evaluation script:
        ```bash
        uv run python evaluate_rag.py
        ```
        3. Refresh this page to see results
        """)
    
    st.stop()

with st.sidebar:
    st.markdown("### Select Evaluation Run")
    
    selected_file = st.selectbox(
        "Choose a result file:",
        result_files,
        format_func=lambda x: x.stem.replace("evaluation_results_", "")
    )
    
    st.markdown("---")
    st.markdown("### RAGAS Metrics")
    st.markdown("""
    **Faithfulness**: Answer grounded in context?
    
    **Answer Relevancy**: Answer addresses question?
    """)

with open(selected_file, 'r') as f:
    data = json.load(f)

ragas_scores = data["ragas_scores"]
detailed_results = data["detailed_results"]
summary = data.get("summary", {})

st.markdown("### Overall Metrics")

col1, col2 = st.columns(2)

metrics_info = {
    "faithfulness": ("Faithfulness", col1),
    "answer_relevancy": ("Answer Relevancy", col2)
}

for metric_key, (metric_name, col) in metrics_info.items():
    if metric_key in ragas_scores:
        score = ragas_scores[metric_key]
        with col:
            st.metric(metric_name, f"{score:.3f}", f"{score*100:.1f}%")

avg_score = sum(ragas_scores.values()) / len(ragas_scores)
st.markdown(f"**Average Score:** {avg_score:.3f} ({avg_score*100:.1f}%)")

st.markdown("---")

fig = go.Figure()

fig.add_trace(go.Bar(
    x=list(ragas_scores.keys()),
    y=list(ragas_scores.values()),
    text=[f"{v*100:.1f}%" for v in ragas_scores.values()],
    textposition='auto',
    marker_color=['#667eea', '#764ba2']
))

fig.update_layout(
    title="RAGAS Metrics Overview",
    yaxis_title="Score (0-1)",
    xaxis_title="Metric",
    yaxis=dict(range=[0, 1]),
    height=400,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### Query Details")

for idx, result in enumerate(detailed_results, 1):
    with st.expander(f"Query {idx}: {result['question'][:60]}..."):
        st.markdown(f"**Question:** {result['question']}")
        st.markdown(f"**Answer:** {result['answer']}")
        st.markdown(f"**Ground Truth:** {result['ground_truth']}")
        
        st.markdown("**Retrieved Contexts:**")
        for ctx_idx, context in enumerate(result['contexts'], 1):
            st.markdown(f"{ctx_idx}. `{context[:200]}...`")

st.markdown("---")
st.markdown("### Summary Statistics")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Queries", summary.get("total_queries", len(detailed_results)))
    
with col2:
    avg_contexts = summary.get("avg_contexts_per_query", 
                               sum(len(r["contexts"]) for r in detailed_results) / len(detailed_results))
    st.metric("Avg. Contexts per Query", f"{avg_contexts:.1f}")

st.markdown("---")
st.markdown("### Interpretation Guide")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Score Ranges:**
    - 0.8 - 1.0: Excellent
    - 0.6 - 0.8: Good
    - 0.4 - 0.6: Fair
    - 0.0 - 0.4: Needs improvement
    """)

with col2:
    st.markdown("""
    **Recommended Actions:**
    - Low Faithfulness: Reduce temperature, improve prompts
    - Low Relevancy: Better question understanding, improve generation prompts
    """)

st.markdown("---")

with st.expander("Export for CV/Resume"):
    st.markdown("Use these metrics in your CV:")
    
    st.code(f"""
Achieved {ragas_scores.get('faithfulness', 0)*100:.0f}% faithfulness and 
{ragas_scores.get('answer_relevancy', 0)*100:.0f}% answer relevancy 
(RAGAS metrics) on technical documentation Q&A
    """.strip())
    
    st.markdown("**Full metrics statement:**")
    st.code(f"""
Evaluated RAG system with RAGAS framework achieving:
- Faithfulness: {ragas_scores.get('faithfulness', 0)*100:.1f}%
- Answer Relevancy: {ragas_scores.get('answer_relevancy', 0)*100:.1f}%
Average: {avg_score*100:.1f}%
    """.strip())