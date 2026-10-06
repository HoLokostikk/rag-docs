from src.search_hybrid import search_hybrid

for prefetch in [10, 30, 50, 100]:
    c = search_hybrid("how many attention heads in the base configuration",
                      "rag_clean_md", top_k=prefetch)
    print(f"prefetch={prefetch:3d} → {len(c)} кандидатів")