"""A chat UI over ragcore: upload documents, ask follow-up questions, see the cited passages.

    uv sync --extra local --extra examples
    uv run streamlit run examples/streamlit_app.py

Uses ``profiles/local.toml`` (set RAGCORE_PROFILE to use another). The generator in the profile must be
running; without it you can still ingest and inspect what retrieval returns.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from ragcore import ConversationState, build_rag, load_profile
from ragcore.loaders.files import FileLoader

PROFILE = os.getenv("RAGCORE_PROFILE", "profiles/local.toml")

st.set_page_config(page_title="ragcore", layout="centered")


@st.cache_resource(show_spinner="Loading models…")
def pipeline():
    return build_rag(load_profile(PROFILE))


rag = pipeline()
if "state" not in st.session_state:
    st.session_state.state = ConversationState()
    st.session_state.shown = []          # [(question, Answer)]

with st.sidebar:
    st.header("Documents")
    st.caption(f"{rag.store.count()} passages stored · profile `{PROFILE}`")
    uploads = st.file_uploader("Add PDF, Markdown or text files", type=["pdf", "md", "txt"],
                               accept_multiple_files=True)
    if uploads and st.button("Ingest"):
        with tempfile.TemporaryDirectory() as folder:
            for upload in uploads:
                (Path(folder) / upload.name).write_bytes(upload.getbuffer())
            report = rag.ingest(FileLoader(folder))
        st.success(f"{report.ingested} ingested ({report.chunks} passages), {report.skipped} unchanged")
    if st.button("New conversation"):
        st.session_state.state = ConversationState()
        st.session_state.shown = []
        st.rerun()

st.title("Ask your documents")
generator_ready = rag.generator is not None and getattr(rag.generator, "available", lambda: True)()
if not generator_ready:
    st.warning("The generator in this profile is not answering, so questions show the retrieved passages only.")


def show(answer) -> None:
    if answer.standalone_question and answer.standalone_question != answer.question:
        st.caption(f"Searched for: {answer.standalone_question}")
    st.write(answer.text)
    if answer.abstained and answer.reason:
        st.caption(f"Refused: {answer.reason.replace('_', ' ')}")
    for citation in answer.citations:
        hit = answer.hits[citation.n - 1]
        with st.expander(f"[{citation.n}] {citation.source} · {hit.chunk.metadata.get('section', '')}"):
            st.write(hit.chunk.text)


for question, answer in st.session_state.shown:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        show(answer)

if message := st.chat_input("Ask a question"):
    st.chat_message("user").write(message)
    with st.chat_message("assistant"):
        if generator_ready:
            answer = rag.chat(st.session_state.state, message)
            show(answer)
            st.session_state.shown.append((message, answer))
        else:
            for n, hit in enumerate(rag.retrieve(message), start=1):
                with st.expander(f"{n}. {hit.chunk.doc_id} · score {hit.score:.3f}", expanded=n == 1):
                    st.write(hit.chunk.text)
