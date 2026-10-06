
import argparse

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def search(query: str, collection: str = "rag_headings", top_k: int = 5) -> list[dict]:
    model = get_model()
    client = QdrantClient(url="http://localhost:6333")

    vector = model.encode(f"query: {query}", normalize_embeddings=True)

    hits = client.query_points(
        collection_name=collection,
        query=vector.tolist(),
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "score": hit.score,
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "index": hit.payload["index"],
        }
        for hit in hits
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--collection", default="rag_headings")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    results = search(args.query, args.collection, args.top_k)

    print(f"query: {args.query}")
    print(f"collection: {args.collection}\n")

    for i, r in enumerate(results, 1):
        preview = r["text"] if args.full else r["text"][:300].replace("\n", " ")
        print(f"--- {i}. score={r['score']:.4f}  {r['source']}#{r['index']} ---")
        print(preview)
        print()