
import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("index", type=int)
    parser.add_argument("--collection", default="rag_clean_md")
    args = parser.parse_args()

    client = QdrantClient(url="http://localhost:6333")

    points, _ = client.scroll(
        args.collection,
        scroll_filter=Filter(must=[
            FieldCondition(key="source", match=MatchValue(value=args.source)),
            FieldCondition(key="index", match=MatchValue(value=args.index)),
        ]),
        limit=1,
        with_payload=True,
    )

    if not points:
        print(f"не знайдено: {args.source}#{args.index}")
        return

    p = points[0].payload
    print(f"=== {p['source']}#{p['index']} ({len(p['text'])} chars) ===")
    print(f"heading: {p.get('heading', '—')}\n")
    print(p["text"])


if __name__ == "__main__":
    main()