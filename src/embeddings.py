"""LangChain `Embeddings` backed by a local sentence-transformers model.

Runs on CPU. The model is loaded once per process and cached, because re-loading it for every
`embed_query` call would dominate retrieval latency (the model load is ~1 s, a batch of 100
chunks is ~0.3 s).
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL, device="cpu")


class MiniLMEmbeddings(Embeddings):
    """384-dimensional, L2-normalised embeddings from all-MiniLM-L6-v2.

    Normalising here lets FAISS use inner product as cosine similarity, which is what
    `IndexFlatIP` + `normalize_L2` gives us in `ingest.py`.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBED_MODEL

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = _model().encode(list(texts), batch_size=32, normalize_embeddings=True,
                               show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return _model().encode([text], normalize_embeddings=True,
                               show_progress_bar=False)[0].tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> MiniLMEmbeddings:
    return MiniLMEmbeddings()
