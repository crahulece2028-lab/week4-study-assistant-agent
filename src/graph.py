"""Task 4.2: the multi-step agent, as a LangGraph state machine.

Shape of the run:

    START -> router -+-> retrieve -> grade -+-> generate -> END
                     |                      |
                     |                      +-> rewrite -> retrieve   (bounded retry)
                     +-> tool ------> generate -> END
                     +-> direct ----> END

Why a graph and not a chain: the control flow is *conditional and cyclic*. Whether a question
touches the notes at all is decided at runtime; whether one retrieval is enough is decided by a
grader node; and a failed grade loops back through query rewriting. That loop is the thing a linear
chain cannot express, and it is the mechanism task 4.3 evaluates.

State is a TypedDict. Each node returns only the keys it changes. `trace` accumulates one entry per
node with wall-clock timing, which `eval/local_tracer.py` turns into a waterfall when LangSmith is
not configured.
"""
from __future__ import annotations

import os
import time
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

import config
import rag
import tools
from llm import get_llm

MAX_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "2"))
# If the grader sees almost no lexical overlap between question and context, the topic is probably
# simply absent from the corpus, and rewriting the query will not conjure it up. Below this floor
# we skip the retry loop and answer with low confidence immediately. See eval/improvement_experiment.py
# for the measured effect (it removes ~2 redundant retrievals on unanswerable questions).
LOW_COVER_FLOOR = float(os.getenv("RAG_LOW_COVER_FLOOR", "0.15"))


class AgentState(TypedDict, total=False):
    question: str
    query: str                 # current retrieval query (starts == question, may be rewritten)
    route: str                 # retrieve | tool | direct
    route_reason: str
    tool_name: str
    tool_input: str
    tool_result: str
    tool_ok: bool
    docs: list[dict]           # retrieval hits (dicts, see rag.retrieve_docs)
    grade_relevant: bool
    grade_score: float
    grade_reason: str
    retry_count: int
    answer: str
    sources: list[str]
    confidence: str            # grounded | low | tool | direct
    trace: Annotated[list[dict], lambda a, b: a + b]


def _step(node: str, detail: str, elapsed_ms: float, **extra) -> dict:
    entry = {"node": node, "detail": detail, "elapsed_ms": round(elapsed_ms, 1)}
    entry.update(extra)
    return entry


def make_nodes(model):
    """Build the node callables bound to a specific LLM (offline or a real provider)."""

    def router(state: AgentState) -> dict:
        t0 = time.perf_counter()
        decision = model.route(state["question"])
        return {
            "route": decision.route,
            "route_reason": decision.reason,
            "tool_name": decision.tool_name or "",
            "tool_input": decision.tool_input or "",
            "query": state.get("query") or state["question"],
            "retry_count": state.get("retry_count", 0),
            "trace": [_step("router", f"-> {decision.route}: {decision.reason}",
                            (time.perf_counter() - t0) * 1000)],
        }

    def retrieve(state: AgentState) -> dict:
        t0 = time.perf_counter()
        query = state.get("query") or state["question"]
        docs = rag.retrieve_docs(query, k=config.TOP_K)
        top = ", ".join(f"{d['source']}({d['score']:.2f})" for d in docs[:3])
        return {
            "docs": docs,
            "trace": [_step("retrieve", f"k={config.TOP_K} for '{query[:60]}' -> {top}",
                            (time.perf_counter() - t0) * 1000,
                            hits=[{"source": d["source"], "score": d["score"]} for d in docs])],
        }

    def grade(state: AgentState) -> dict:
        t0 = time.perf_counter()
        docs = [(d["content"], d["source"]) for d in state.get("docs", [])]
        g = model.grade(state["question"], docs)
        return {
            "grade_relevant": g.relevant,
            "grade_score": g.score,
            "grade_reason": g.reason,
            "trace": [_step("grade", f"relevant={g.relevant} ({g.reason})",
                            (time.perf_counter() - t0) * 1000)],
        }

    def rewrite(state: AgentState) -> dict:
        t0 = time.perf_counter()
        docs = [(d["content"], d["source"]) for d in state.get("docs", [])]
        new_query = model.rewrite(state["question"], docs)
        n = state.get("retry_count", 0) + 1
        return {
            "query": new_query,
            "retry_count": n,
            "trace": [_step("rewrite", f"retry {n}: '{new_query[:70]}'",
                            (time.perf_counter() - t0) * 1000)],
        }

    def tool(state: AgentState) -> dict:
        t0 = time.perf_counter()
        result = tools.run_tool(state.get("tool_name", ""), state.get("tool_input", ""))
        return {
            "tool_result": result.text,
            "tool_ok": result.ok,
            "trace": [_step("tool", f"{result.name}({state.get('tool_input','')}) ok={result.ok}",
                            (time.perf_counter() - t0) * 1000)],
        }

    def generate(state: AgentState) -> dict:
        t0 = time.perf_counter()
        docs = [(d["content"], d["source"]) for d in state.get("docs", [])]
        route = type("R", (), {"route": state.get("route", "retrieve"),
                               "tool_name": state.get("tool_name"),
                               "tool_input": state.get("tool_input")})
        low = (state.get("route") == "retrieve" and not state.get("grade_relevant", False))
        answer = model.generate(state["question"], docs, route,
                                tool_result=state.get("tool_result"),
                                low_confidence=low)
        if state.get("route") == "tool":
            confidence = "tool"
        elif state.get("route") == "direct":
            confidence = "direct"
        elif low:
            confidence = "low"
        else:
            confidence = "grounded"
        sources = sorted({d["source"] for d in state.get("docs", [])}) if docs else []
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "trace": [_step("generate", f"{confidence} answer, {len(answer)} chars",
                            (time.perf_counter() - t0) * 1000)],
        }

    def direct(state: AgentState) -> dict:
        t0 = time.perf_counter()
        route = type("R", (), {"route": "direct", "tool_name": None, "tool_input": None})
        answer = model.generate(state["question"], [], route)
        return {
            "answer": answer,
            "sources": [],
            "confidence": "direct",
            "trace": [_step("direct", "answered without retrieval",
                            (time.perf_counter() - t0) * 1000)],
        }

    return {"router": router, "retrieve": retrieve, "grade": grade,
            "rewrite": rewrite, "tool": tool, "generate": generate, "direct": direct}


def _route_after_router(state: AgentState) -> str:
    return state.get("route", "retrieve")


def _route_after_grade(state: AgentState) -> str:
    if state.get("grade_relevant", False):
        return "generate"
    if state.get("grade_score", 0.0) < LOW_COVER_FLOOR:
        return "generate"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite"
    return "generate"


def build_graph(model=None):
    model = model or get_llm()
    nodes = make_nodes(model)
    g = StateGraph(AgentState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", _route_after_router,
                            {"retrieve": "retrieve", "tool": "tool", "direct": "direct"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _route_after_grade,
                            {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("tool", "generate")
    g.add_edge("generate", END)
    g.add_edge("direct", END)
    return g.compile(checkpointer=MemorySaver())


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def answer(question: str, thread_id: str = "default") -> AgentState:
    """Run one question through the graph and return the final state."""
    graph = get_graph()
    init: AgentState = {"question": question, "query": question, "retry_count": 0, "trace": []}
    return graph.invoke(init, config={"configurable": {"thread_id": thread_id}})


def mermaid() -> str:
    """Graph topology as Mermaid text. Requires no network, unlike draw_mermaid_png()."""
    return get_graph().get_graph().draw_mermaid()


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the Nyquist rate?"
    final = answer(q)
    print(f"Q: {q}")
    print(f"route: {final.get('route')} | confidence: {final.get('confidence')}")
    print(f"A: {final.get('answer')}")
    print(f"sources: {final.get('sources')}")
    print("trace:")
    for step in final.get("trace", []):
        print(f"  - {step['node']:9s} {step['elapsed_ms']:7.1f}ms  {step['detail']}")
