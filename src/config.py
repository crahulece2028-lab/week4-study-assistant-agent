"""Central configuration. Resolves paths and decides the LLM mode from the environment.

Every other module imports from here rather than reading `os.environ` itself, so there is exactly
one place that decides whether the system talks to Groq, Gemini, or the offline extractive model.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INDEX = ROOT / "index"
DOCS = ROOT / "docs"
EVAL = ROOT / "eval"

load_dotenv(ROOT / ".env")

# --- retrieval defaults (CLI flags override at call sites) ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

LANGSMITH_API_KEY = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "week4-study-assistant")


def llm_mode() -> str:
    """Which LLM backend is active. Offline means 'no key was provided', not 'broken'."""
    if GROQ_API_KEY:
        return "groq"
    if GOOGLE_API_KEY:
        return "gemini"
    return "offline"


def tracing_enabled() -> bool:
    return bool(LANGSMITH_API_KEY) and os.getenv("LANGSMITH_TRACING", "false").lower() == "true"


def enable_langsmith() -> bool:
    """Turn on LangSmith env vars *before* langchain is imported anywhere.

    If no key is present this is a no-op, and the local tracer in eval/local_tracer.py records an
    equivalent span tree to disk instead, so the evaluation still produces trace artifacts.
    """
    if not tracing_enabled():
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    return True


def describe() -> dict:
    return {
        "llm_mode": llm_mode(),
        "groq_model": GROQ_MODEL if GROQ_API_KEY else None,
        "google_model": GOOGLE_MODEL if GOOGLE_API_KEY else None,
        "langsmith_tracing": tracing_enabled(),
        "langsmith_project": LANGSMITH_PROJECT if tracing_enabled() else None,
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
    }
