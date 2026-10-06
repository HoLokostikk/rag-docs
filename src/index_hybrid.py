
import argparse
import uuid
from pathlib import Path

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from src.chunking import STRATEGIES, is_noise, clean_text

PROCESSED_DIR = Path("data/processed")
DENSE_MODEL = "intfloat/multilingual-e5-small"
SPARSE_MODEL = "Qdrant/bm25"
MIN_CHUNK_CHARS = 100


def load_chunks(strategy: str) -> list:
    chunk_fn = STRATEGIES[strategy]
    all_chunks = []

    for path in sorted(PROCESSED_DIR.glob("*.md")):
        chunks = chunk_fn(path.read_text(encoding="utf-8"), source=path.stem)
        kept = [c for c in chunks
                if c.char_len >= MIN_CHUNK_CHARS and not is_noise(c)]
        for c in kept:
            c.text = clean_text(c.text)
        all_chunks.extend(kept)
        print(f"{path.stem:12s} {len(kept):4d} chunks (of {len(chunks)})")

    print(f"\ntotal: {len(all_chunks)}")
    return all_chunks


def index(strategy: str, collection: str):
    chunks = load_chunks(strategy)

    print(f"\nloading models...")
    dense_model = SentenceTransformer(DENSE_MODEL)
    sparse_model = SparseTextEmbedding(SPARSE_MODEL)
    dim = dense_model.get_embedding_dimension()

    client = QdrantClient(url="http://localhost:6333")

    if client.collection_exists(collection):
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": VectorParams(size=dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(index=SparseIndexParams()),
        },
    )

    texts = [c.text for c in chunks]

    print("embedding (dense)...")
    dense_vectors = dense_model.encode(
        [f"passage: {t}" for t in texts],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("embedding (sparse)...")
    sparse_vectors = list(sparse_model.embed(texts))

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense.tolist(),
                "bm25": SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
            },
            payload={
                "text": chunk.text,
                "source": chunk.source,
                "index": chunk.index,
                **chunk.meta,
            },
        )
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
    ]

    client.upsert(collection_name=collection, points=points)

    info = client.get_collection(collection)
    print(f"\ncollection '{collection}': {info.points_count} points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="headings", choices=list(STRATEGIES))
    parser.add_argument("--collection", default="rag_hybrid")
    args = parser.parse_args()

    index(args.strategy, args.collection)