"""Task 4.1, stage 2 of 2: the retrieval layer.

A thin, opinionated wrapper over the FAISS store so the rest of the code never touches vector
internals. Because the index is `IndexFlatIP` over L2-normalised vectors, the score returned here
*is* cosine similarity in [-1, 1], and higher is better. `retrieve_docs` returns a dict per hit so
callers (the graph, the eval harness, the demo) share one shape.
"""
from __future__ import annotations

from functools import lru_cache

import config
from store import load_index


@lru_cache(maxsize=1)
def get_store():
    return load_index()


def retrieve_docs(question: str, k: int | None = None) -> list[dict]:
    k = k or config.TOP_K
    hits = get_store().similarity_search_with_score(question, k=k)
    out = []
    for doc, score in hits:
        out.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "?"),
            "page": doc.metadata.get("page", 0),
            "chunk_id": doc.metadata.get("chunk_id"),
            "score": round(float(score), 4),
        })
    return out


def retrieve_pairs(question: str, k: int | None = None) -> list[tuple[str, str]]:
    """The (text, source) shape the LLM layer consumes."""
    return [(h["content"], h["source"]) for h in retrieve_docs(question, k)]


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{h['source']} p{h['page']}] {h['content']}" for h in hits)
