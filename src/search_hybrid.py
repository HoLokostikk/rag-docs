
import argparse

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import FusionQuery, Prefetch, SparseVector
from sentence_transformers import SentenceTransformer

DENSE_MODEL = "intfloat/multilingual-e5-small"
SPARSE_MODEL = "Qdrant/bm25"

_dense = None
_sparse = None


def models():
    global _dense, _sparse
    if _dense is None:
        _dense = SentenceTransformer(DENSE_MODEL)
        _sparse = SparseTextEmbedding(SPARSE_MODEL)
    return _dense, _sparse


def search_dense(query: str, collection: str, top_k: int = 5) -> list[dict]:
    dense, _ = models()
    client = QdrantClient(url="http://localhost:6333")

    vector = dense.encode(f"query: {query}", normalize_embeddings=True)
    hits = client.query_points(
        collection_name=collection,
        query=vector.tolist(),
        using="dense",
        limit=top_k,
        with_payload=True,
    ).points
    return [_fmt(h) for h in hits]


def search_bm25(query: str, collection: str, top_k: int = 5) -> list[dict]:
    _, sparse = models()
    client = QdrantClient(url="http://localhost:6333")

    vec = list(sparse.embed([query]))[0]
    hits = client.query_points(
        collection_name=collection,
        query=SparseVector(indices=vec.indices.tolist(),
                           values=vec.values.tolist()),
        using="bm25",
        limit=top_k,
        with_payload=True,
    ).points
    return [_fmt(h) for h in hits]


def search_hybrid(query: str, collection: str, top_k: int = 5,
                  prefetch: int | None = None) -> list[dict]:
    if prefetch is None:
        prefetch = max(20, top_k * 2)

    dense, sparse = models()
    client = QdrantClient(url="http://localhost:6333")

    dense_vec = dense.encode(f"query: {query}", normalize_embeddings=True)
    sparse_vec = list(sparse.embed([query]))[0]

    hits = client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(query=dense_vec.tolist(), using="dense", limit=prefetch),
            Prefetch(
                query=SparseVector(indices=sparse_vec.indices.tolist(),
                                   values=sparse_vec.values.tolist()),
                using="bm25",
                limit=prefetch,
            ),
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k,
        with_payload=True,
    ).points
    return [_fmt(h) for h in hits]


def _fmt(hit) -> dict:
    return {
        "score": hit.score,
        "text": hit.payload["text"],
        "source": hit.payload["source"],
        "index": hit.payload["index"],
    }


def show(title: str, results: list[dict]):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    for i, r in enumerate(results, 1):
        preview = r["text"][:180].replace("\n", " ")
        print(f"{i}. [{r['score']:.4f}] {r['source']}#{r['index']}")
        print(f"   {preview}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--collection", default="rag_clean_md")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(f"query: {args.query}")
    print(f"collection: {args.collection}")

    show("DENSE", search_dense(args.query, args.collection, args.top_k))
    show("BM25", search_bm25(args.query, args.collection, args.top_k))
    show("HYBRID (RRF)", search_hybrid(args.query, args.collection, args.top_k))