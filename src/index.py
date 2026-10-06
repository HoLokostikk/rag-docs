
import argparse
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.chunking import STRATEGIES, is_noise, clean_text


PROCESSED_DIR = Path("data/processed")
MODEL_NAME = "intfloat/multilingual-e5-small"
MIN_CHUNK_CHARS = 100


def load_chunks(strategy : str) -> list:
    chunk_fn = STRATEGIES[strategy]
    all_chunks = []
    dropped = 0

    for path in sorted(PROCESSED_DIR.glob("*.md")):
        text = path.read_text(encoding = "utf-8")

        chunks = chunk_fn(text, source = path.stem)

        kept = [c for c in chunks
                if c.char_len >= MIN_CHUNK_CHARS and not is_noise(c)]
        dropped_short = sum(1 for c in chunks if c.char_len < MIN_CHUNK_CHARS)
        dropped_noise = sum(1 for c in chunks if is_noise(c))
        for c in kept:
            c.text = clean_text(c.text)
        print(f"{path.stem:12s} {len(kept):4d} chunks "
              f"(-{dropped_short} short, -{dropped_noise} noise)")
        all_chunks.extend(kept)
        print(f"{path.stem:12s} {len(kept):4d} chunks "
              f"({len(chunks) - len(kept)} dropped as too short)")

        print(f"\ntotal: {len(all_chunks)} chunks, {dropped} dropped")
    return all_chunks


def index(strategy: str, collection: str, batch_size: int = 32):
    chunks = load_chunks(strategy)

    print(f"\nloading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_embedding_dimension()

    client = QdrantClient(url="http://localhost:6333")

    if client.collection_exists(collection):
        print(f"recreating collection {collection}")
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    print(f"embedding {len(chunks)} chunks...")
    texts = [f"passage: {c.text}" for c in chunks]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={
                "text": chunk.text,
                "source": chunk.source,
                "index": chunk.index,
                "strategy": strategy,
                "char_len": chunk.char_len,
                **chunk.meta,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(collection_name=collection, points=points)

    info = client.get_collection(collection)
    print(f"\ncollection '{collection}': {info.points_count} points, dim={dim}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="headings", choices=list(STRATEGIES))
    parser.add_argument("--collection")
    args = parser.parse_args()

    collection = args.collection or f"rag_{args.strategy}"
    index(args.strategy, collection)