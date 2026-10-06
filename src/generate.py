

import argparse

import httpx

from src.rerank import search_with_rerank

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:3b-instruct-q4_K_M"
COLLECTION = "rag_clean_md"

SYSTEM = """You answer questions using only the provided context.

Rules:
- Use only information from the context. Do not use prior knowledge.
- Every factual claim must cite its source as [1], [2], etc.
- If the context does not contain the answer, reply exactly:
  "The provided context does not contain this information."
- Be concise. Two or three sentences unless the question requires more.
- Before answering, check that the cited source actually discusses the
  subject of the question. If the context describes a different model
  or system, say so instead of answering."""

USER_TEMPLATE = """Context:
{context}

Question: {question}"""


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (source: {c['source']}#{c['index']})\n{c['text']}")
    return "\n\n".join(parts)


def answer(question: str, collection: str = COLLECTION, top_k: int = 3,
           prefetch: int = 10) -> dict:
    chunks = search_with_rerank(question, collection, top_k, prefetch)
    context = format_context(chunks)

    response = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TEMPLATE.format(
                    context=context, question=question)},
            ],
            "stream": False,
            "options": {"temperature": 0, "seed": 0},
        },
        timeout=600.0,
    )
    response.raise_for_status()
    data = response.json()

    return {
        "question": question,
        "answer": data["message"]["content"].strip(),
        "chunks": chunks,
        "context": context,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--prefetch", type=int, default=10)
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args()

    result = answer(args.question, args.collection, args.top_k, args.prefetch)

    if args.show_context:
        print("=" * 70)
        print(result["context"])
        print("=" * 70)

    print(f"\nQ: {result['question']}\n")
    print(result["answer"])

    sources = [f"{c['source']}#{c['index']}" for c in result["chunks"]]
    print(f"\nsources: {sources}")
    print(f"tokens: {result['prompt_tokens']} in, {result['completion_tokens']} out")