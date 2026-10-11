"""Persistence for the vector store.

`FAISS.save_local` / `load_local` delegate to `faiss.write_index` / `read_index`, whose C++ file
APIs take a `const char*` and therefore use the Windows ANSI code page. This project lives under a
path containing non-ASCII characters, so those calls raise `could not open ...: No such file or
directory` even though the directory exists.

`serialize_to_bytes` / `deserialize_from_bytes` avoid the C++ file layer entirely: faiss serialises
to a `bytes` buffer and Python (which is Unicode-aware) does the actual disk I/O. The resulting file
is a pickle, hence `allow_dangerous_deserialization=True` on load — acceptable here because the file
is produced by `ingest.py` in the same checkout and is git-ignored.
"""
from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

import config
from embeddings import get_embeddings

STORE_FILE = config.INDEX / "faiss_store.bin"

_INDEX_KWARGS = dict(
    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    normalize_L2=True,
)


def save_index(store: FAISS) -> None:
    config.INDEX.mkdir(parents=True, exist_ok=True)
    STORE_FILE.write_bytes(store.serialize_to_bytes())


def load_index() -> FAISS:
    if not STORE_FILE.exists():
        raise SystemExit(f"no index at {STORE_FILE}\nRun: python src/ingest.py")
    data = STORE_FILE.read_bytes()
    return FAISS.deserialize_from_bytes(
        data,
        get_embeddings(),
        allow_dangerous_deserialization=True,
        **_INDEX_KWARGS,
    )
