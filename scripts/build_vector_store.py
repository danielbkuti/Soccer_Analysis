"""
Build a Chroma vector store from the RAG corpus (data/processed/match_notes.jsonl).

Embeds each match note with a local sentence-transformers model and persists
the resulting vectors + metadata to disk (data/processed/chroma_db/), so later
scripts can query it without re-embedding every time. Re-running this script
rebuilds the collection from scratch, so it stays in sync after the corpus is
regenerated.

Usage:
    python scripts/build_vector_store.py
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "processed" / "match_notes.jsonl"
STORE_PATH = ROOT / "data" / "processed" / "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "match_notes"


def load_corpus():
    docs = []
    with CORPUS_PATH.open() as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def main():
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents from {CORPUS_PATH}")

    print(f"Loading embedding model: {MODEL_NAME} (downloads once, ~90MB, then cached)")
    model = SentenceTransformer(MODEL_NAME)

    texts = [d["text"] for d in docs]
    print(f"Embedding {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=str(STORE_PATH))
    # Rebuild fresh each run, so re-running after regenerating the corpus
    # doesn't leave stale documents behind alongside the new ones.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[d["id"] for d in docs],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "date": d["date"],
                "season": d["season"],
                "opponent": d["opponent"],
                "is_home": d["is_home"],
                "corners": d["corners"],
            }
            for d in docs
        ],
    )

    print(f"Indexed {collection.count()} documents into '{COLLECTION_NAME}' at {STORE_PATH}")

    # Sanity check: run one real semantic query and show what comes back —
    # this is the actual point of building a vector store, so prove it works.
    query = "a match where City struggled against a defensively strong opponent"
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    print(f"\nSanity check - query: {query!r}")
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"\n  [distance={dist:.4f}] {meta['date']} vs {meta['opponent']} ({meta['corners']} corners)")
        print(f"  {doc}")


if __name__ == "__main__":
    main()
