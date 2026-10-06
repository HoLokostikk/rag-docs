"""
Вплив batch_size на вартість reranking.

Запуск:
    python -m src.bench_rerank
"""

import time

from src.rerank import rerank
from src.search_hybrid import search_hybrid

COLLECTION = "rag_clean_md"
QUERY = "how many attention heads in the base configuration"
RUNS = 3


def measure(fn, *args, runs: int = RUNS, **kwargs) -> float:
    """Середній час у мс, без першого (холодного) виклику."""
    fn(*args, **kwargs)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return 1000 * sum(times) / len(times)


if __name__ == "__main__":
    candidates = search_hybrid(QUERY, COLLECTION, top_k=28)
    print(f"query: {QUERY}")
    print(f"кандидатів: {len(candidates)}\n")

    print(f"{'batch_size':>11s} {'час ms':>9s} {'ms на пару':>12s}")
    print("-" * 34)

    for bs in [1, 2, 4, 8, 16, 32]:
        t = measure(rerank, QUERY, candidates, 5, batch_size=bs)
        print(f"{bs:>11d} {t:>9.1f} {t / len(candidates):>12.1f}")