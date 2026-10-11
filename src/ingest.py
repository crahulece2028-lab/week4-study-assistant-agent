"""Task 4.1, stage 1 of 2: load, chunk, embed and index the notes.

    python src/ingest.py [--chunk-size 800] [--chunk-overlap 150]

Writes a FAISS index to `index/` and a machine-readable summary to `docs/ingest_stats.json`.
The stats file is written by the run rather than by hand, so the architecture notes quote numbers
that cannot drift from the index they describe.
"""
from __future__ import annotations

import argparse
import json
from statistics import mean

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from embeddings import get_embeddings
from store import save_index


def load_documents() -> list:
    pdfs = sorted(config.DATA.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs in {config.DATA}\nRun: python data/build_corpus.py")
    docs = []
    for pdf in pdfs:
        loaded = PyPDFLoader(str(pdf)).load()
        for d in loaded:
            d.metadata["source"] = pdf.stem
            d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
        docs.extend(loaded)
    return docs


def build(chunk_size: int, chunk_overlap: int) -> dict:
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i

    vectors = FAISS.from_documents(
        chunks,
        get_embeddings(),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=True,
    )
    save_index(vectors)

    lengths = [len(c.page_content) for c in chunks]
    stats = {
        "documents": len({d.metadata["source"] for d in docs}),
        "pages": len(docs),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_chars_min": min(lengths),
        "chunk_chars_mean": round(mean(lengths), 1),
        "chunk_chars_max": max(lengths),
        "embedding_model": config.EMBED_MODEL,
        "embedding_dim": len(get_embeddings().embed_query("dimension probe")),
        "index_type": "FAISS IndexFlatIP (cosine via L2-normalised vectors)",
        "documents_list": sorted({d.metadata["source"] for d in docs}),
        "example_chunks": [
            {"source": c.metadata["source"], "page": c.metadata["page"],
             "chars": len(c.page_content),
             "preview": c.page_content[:220].replace("\n", " ")}
            for c in chunks[:3]
        ],
    }
    (config.DOCS / "ingest_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    args = ap.parse_args()

    stats = build(args.chunk_size, args.chunk_overlap)
    print(f"indexed {stats['chunks']} chunks from {stats['documents']} documents "
          f"({stats['pages']} pages)")
    print(f"chunk chars: min {stats['chunk_chars_min']} / mean {stats['chunk_chars_mean']} / "
          f"max {stats['chunk_chars_max']}")
    print(f"embedding: {stats['embedding_model']} ({stats['embedding_dim']}-d)")
    print(f"saved -> {config.INDEX}")


if __name__ == "__main__":
    main()
