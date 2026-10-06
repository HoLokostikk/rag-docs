

import argparse
import time
from pathlib import Path

import yaml

from src.rerank import rerank
from src.search_hybrid import search_bm25, search_dense, search_hybrid

DATASET = Path("datasets/retrieval.yaml")
COLLECTION = "rag_clean_md"


def load_dataset() -> list[dict]:
    return yaml.safe_load(DATASET.read_text(encoding="utf-8"))


def is_relevant(result: dict, relevant: list[dict]) -> bool:
    return any(r["source"] == result["source"] and r["index"] == result["index"]
               for r in relevant)


def first_hit_rank(results: list[dict], relevant: list[dict]) -> int | None:
    """Позиція першого релевантного (1-based) або None."""
    for i, r in enumerate(results, 1):
        if is_relevant(r, relevant):
            return i
    return None


def evaluate(name: str, search_fn, dataset: list[dict], k: int) -> dict:
    hits = 0
    reciprocal_ranks = []
    failures = []
    total_time = 0.0

    for case in dataset:
        t0 = time.perf_counter()
        results = search_fn(case["query"], k)
        total_time += time.perf_counter() - t0

        rank = first_hit_rank(results, case["relevant"])

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
            failures.append({
                "id": case["id"],
                "query": case["query"],
                "expected": case["relevant"],
                "got": [f"{r['source']}#{r['index']}" for r in results],
            })

    n = len(dataset)
    return {
        "name": name,
        "recall": hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "avg_ms": 1000 * total_time / n,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--prefetch", type=int, default=28)
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset()
    k = args.k

    methods = {
        "dense": lambda q, k: search_dense(q, COLLECTION, k),
        "bm25": lambda q, k: search_bm25(q, COLLECTION, k),
        "hybrid": lambda q, k: search_hybrid(q, COLLECTION, k),
        "hybrid+rerank": lambda q, k: rerank(
            q, search_hybrid(q, COLLECTION, args.prefetch), k, batch_size=1
        ),
    }

    print(f"dataset: {len(dataset)} queries, k={k}, prefetch={args.prefetch}\n")

    results = []
    for name, fn in methods.items():
        print(f"running {name}...", flush=True)
        results.append(evaluate(name, fn, dataset, k))

    print(f"\n{'method':16s} {'recall@' + str(k):>9s} {'MRR':>7s} {'ms/query':>9s}")
    print("-" * 45)
    for r in results:
        print(f"{r['name']:16s} {r['recall']:>9.2f} {r['mrr']:>7.3f} "
              f"{r['avg_ms']:>9.0f}")

    if args.show_failures:
        for r in results:
            if not r["failures"]:
                continue
            print(f"\n=== failures: {r['name']} ===")
            for f in r["failures"]:
                exp = [f"{x['source']}#{x['index']}" for x in f["expected"]]
                print(f"  {f['id']}")
                print(f"    query:    {f['query']}")
                print(f"    expected: {exp}")
                print(f"    got:      {f['got']}")


if __name__ == "__main__":
    main()