from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

for name in ["rag_clean", "rag_hybrid", "rag_headings"]:
    if not client.collection_exists(name):
        print(f"{name}: НЕМАЄ\n")
        continue

    info = client.get_collection(name)
    print(f"{name}: {info.points_count} points, status={info.status}")

    points, _ = client.scroll(name, limit=1, with_vectors=True, with_payload=True)
    if not points:
        print("  порожньо\n")
        continue

    p = points[0]
    print(f"  source={p.payload.get('source')} index={p.payload.get('index')}")

    v = p.vector
    if isinstance(v, dict):
        for key, vec in v.items():
            if hasattr(vec, "indices"):
                print(f"  {key}: sparse, {len(vec.indices)} terms")
            else:
                norm = sum(x * x for x in vec) ** 0.5
                print(f"  {key}: dense len={len(vec)} norm={norm:.4f}")
    else:
        norm = sum(x * x for x in v) ** 0.5
        print(f"  unnamed: len={len(v)} norm={norm:.4f}")
    print()