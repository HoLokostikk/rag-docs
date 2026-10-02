
import argparse

from sentence_transformers import CrossEncoder

from src.search_hybrid import search_bm25, search_dense, search_hybrid

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 5,
           batch_size: int = 32) -> list[dict]:

    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs, batch_size=batch_size)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


def search_with_rerank(query: str, collection: str, top_k: int = 5,
                       prefetch: int = 30) -> list[dict]:
    candidates = search_hybrid(query, collection, top_k=prefetch)
    return rerank(query, candidates, top_k)


def show(title: str, results: list[dict], score_key: str = "score"):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    for i, r in enumerate(results, 1):
        preview = r["text"][:170].replace("\n", " ")
        score = r.get(score_key, r["score"])
        print(f"{i}. [{score:+.4f}] {r['source']}#{r['index']}")
        print(f"   {preview}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--collection", default="rag_clean_md")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prefetch", type=int, default=30)
    args = parser.parse_args()

    print(f"query: {args.query}")
    print(f"collection: {args.collection}, prefetch={args.prefetch}")

    show("HYBRID (before rerank)",
         search_hybrid(args.query, args.collection, args.top_k))

    show("AFTER RERANK",
         search_with_rerank(args.query, args.collection, args.top_k, args.prefetch),
         score_key="rerank_score")