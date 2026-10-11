"""Task 4.4: the interactive demo.

    streamlit run src/app.py

Shows the full agent trace for every answer (route, retrieval, grade, retries, tool calls), because
the whole point of the exercise is a system whose *process* is inspectable, not just its output.
Backend is chosen at runtime from the environment; with no API key it runs the offline extractive
model and says so in the sidebar instead of failing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import config  # noqa: E402
import graph  # noqa: E402

EXAMPLES = [
    "What is the Nyquist rate and why does it matter?",
    "How does a quarter-wave transformer match a transmission line?",
    "What does the PID control law look like?",
    "What is the formula for VSWR?",
    "calculate 2*pi*50*0.001",
    "Who won the 2018 FIFA World Cup?",
]

CONFIDENCE_UI = {
    "grounded": ("Answerable from the notes", "green"),
    "low": ("Not found in the notes", "orange"),
    "tool": ("Answered by a tool", "blue"),
    "direct": ("Answered directly", "gray"),
}


@st.cache_resource(show_spinner="Compiling the agent graph and loading embeddings...")
def _graph():
    return graph.get_graph()


@st.cache_data
def _ingest_stats():
    path = config.DOCS / "ingest_stats.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def sidebar() -> None:
    st.sidebar.header("System")
    mode = config.llm_mode()
    st.sidebar.write(f"**LLM backend:** `{mode}`")
    if mode == "offline":
        st.sidebar.caption("No `GROQ_API_KEY` / `GOOGLE_API_KEY` set. Running the deterministic "
                           "extractive model: answers are quoted from the notes with citations.")
    else:
        st.sidebar.caption(f"Generative model: `{config.GROQ_MODEL if mode == 'groq' else config.GOOGLE_MODEL}`")
    st.sidebar.write(f"**Embeddings:** `{config.EMBED_MODEL}`")
    st.sidebar.write(f"**Retrieval:** top-{config.TOP_K}, chunk {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}")
    st.sidebar.write(f"**LangSmith tracing:** {'on' if config.tracing_enabled() else 'off (local tracer)'}")

    stats = _ingest_stats()
    if stats:
        st.sidebar.header("Index")
        st.sidebar.write(f"{stats['documents']} documents, {stats['chunks']} chunks")
        st.sidebar.caption(f"chunk chars: mean {stats['chunk_chars_mean']}, "
                           f"max {stats['chunk_chars_max']}")

    with st.sidebar.expander("How the agent decides"):
        st.markdown(
            "1. **router** - notes question, tool call, or small talk?\n"
            "2. **retrieve** - top-k chunks from FAISS (cosine).\n"
            "3. **grade** - does the context actually answer the question?\n"
            "4. **rewrite -> retrieve** - one bounded retry if it does not.\n"
            "5. **generate** - answer grounded in the retrieved chunks.")

    with st.sidebar.expander("Example questions"):
        for q in EXAMPLES:
            st.code(q, language=None)


def render_trace(final: dict) -> None:
    trace = final.get("trace", [])
    if not trace:
        return
    total = sum(s.get("elapsed_ms", 0) for s in trace)
    with st.expander(f"Agent trace ({len(trace)} steps, {total:.1f} ms)"):
        st.table([{"step": i + 1, "node": s["node"], "elapsed_ms": s["elapsed_ms"],
                   "detail": s["detail"]} for i, s in enumerate(trace)])


def render_sources(final: dict) -> None:
    docs = final.get("docs", [])
    if not docs:
        return
    with st.expander(f"Retrieved chunks ({len(docs)})"):
        for d in docs:
            st.markdown(f"**{d['source']}** p{d['page']}  -  cosine `{d['score']}`")
            st.caption(d["content"][:400] + ("..." if len(d["content"]) > 400 else ""))


def main() -> None:
    st.set_page_config(page_title="Study Assistant Agent", page_icon="book", layout="wide")
    st.title("ECE Study Assistant")
    st.caption("A retrieval-augmented agent over eight lecture-note documents. "
               "Ask a question about the notes, do a calculation, or ask for a formula.")
    sidebar()

    if not (config.INDEX / "faiss_store.bin").exists():
        st.error("No index found. Build it first: `python src/ingest.py`")
        return

    if "history" not in st.session_state:
        st.session_state.history = []

    prompt = st.chat_input("Ask about signals, control, comms, DSP, semiconductors...")
    if prompt:
        with st.spinner("Thinking..."):
            final = graph.answer(prompt, thread_id=f"app-{len(st.session_state.history)}")
        st.session_state.history.append({"question": prompt, "final": final})

    for turn in st.session_state.history:
        final = turn["final"]
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            label, color = CONFIDENCE_UI.get(final.get("confidence", ""), ("", "gray"))
            if label:
                st.markdown(f":{color}[**{label}**]")
            st.write(final.get("answer", "(no answer)"))
            if final.get("sources"):
                st.caption("sources: " + ", ".join(final["sources"]))
            render_trace(final)
            render_sources(final)


if __name__ == "__main__":
    _graph()  # warm on first load
    main()
