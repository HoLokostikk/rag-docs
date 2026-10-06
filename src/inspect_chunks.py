

import argparse
import statistics
from pathlib import Path

from src.chunking import STRATEGIES

PROCESSED_DIR = Path("data/processed")


def stats(chunks) -> dict:
    lengths = [c.char_len for c in chunks]
    return {
        "count": len(chunks),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "tiny": sum(1 for x in lengths if x < 100),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="attention")
    parser.add_argument("--strategy")
    parser.add_argument("--show", type=int, default=0)
    parser.add_argument("--find")
    args = parser.parse_args()

    text = (PROCESSED_DIR / f"{args.file}.md").read_text(encoding="utf-8")
    print(f"{args.file}.md: {len(text)} chars\n")

    names = [args.strategy] if args.strategy else list(STRATEGIES)
    results = {}

    print(f"{'strategy':12s} {'count':>6s} {'mean':>7s} {'median':>7s} "
          f"{'min':>6s} {'max':>7s} {'<100':>5s}")
    print("-" * 56)

    for name in names:
        chunks = STRATEGIES[name](text, source=args.file)
        results[name] = chunks
        s = stats(chunks)
        print(f"{name:12s} {s['count']:>6d} {s['mean']:>7.0f} {s['median']:>7.0f} "
              f"{s['min']:>6d} {s['max']:>7d} {s['tiny']:>5d}")

    if args.find:
        print(f"\nchunks containing {args.find!r}:")
        for name in names:
            hits = [c for c in results[name] if args.find in c.text]
            print(f"  {name:12s} {len(hits)} chunks: {[c.index for c in hits]}")

    if args.show:
        for name in names:
            print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
            for chunk in results[name][:args.show]:
                print(f"\n--- chunk {chunk.index} ({chunk.char_len} chars) ---")
                print(chunk.text[:400])
                if chunk.char_len > 400:
                    print("...")


if __name__ == "__main__":
    main()
