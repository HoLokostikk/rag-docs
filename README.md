# rag-docs

A hybrid RAG pipeline over a research-paper corpus, evaluated end to end on a
hand-labelled golden dataset.

The point of the project is not "retrieve chunks and call an LLM" — it is
measuring which parts of the pipeline actually matter, and finding out that
some widely-recommended ones don't.

**Corpus:** Attention Is All You Need, BERT, LoRA (164 chunks after filtering)
**Stack:** Qdrant, sentence-transformers (E5), BM25 sparse vectors, cross-encoder
reranking, vLLM + Qwen2.5-7B for generation and as an LLM judge

---

## Headline results

| metric | value |
|---|---|
| retrieval recall@3 (hybrid + rerank) | **0.94** |
| retrieval recall@10 | **1.00** |
| answer correctness (16 questions, LLM judge) | **0.75** |
| faithfulness | 0.86 |

What moved the numbers, and what didn't:

| change | effect |
|---|---|
| cross-encoder reranking | recall@3 **0.69 → 0.94**, correctness **+0.07…+0.19** |
| markdown cleaning before indexing | BM25 results **2 → 5**, correct chunk from #2 to #1 |
| noise filtering (bibliography) | removed 13% of one document, zero answer content |
| **chunking strategy** | **no significant difference** (see below) |

---

## Pipeline

```
ingest:  PDF → markdown → clean → chunk → filter → dense + sparse vectors → Qdrant
query:   question → hybrid retrieval (RRF) → cross-encoder rerank → top-3 → LLM
```

Two vectors per point in a single Qdrant collection — a dense E5 embedding and a
BM25 sparse vector — fused with Reciprocal Rank Fusion, then reranked.

---

## What the evaluation showed

### Chunking strategy does not matter here

Three strategies compared on end-to-end answer quality, with and without reranking:

| strategy | with rerank | without rerank |
|---|---|---|
| heading-based | 0.69 | 0.62 |
| recursive | **0.75** | 0.56 |
| fixed-size | 0.69 | 0.62 |

A spread of one question out of sixteen. Reranking adds 7–19 points regardless of
how the text was cut.

This contradicts the structural argument — fixed-size splitting visibly cuts
tables in half, leaving header-less rows that are semantically meaningless — and
it contradicts most RAG write-ups, which treat chunking as critical.

The reason is that **every stable failure has a cause orthogonal to chunking**:

- two questions whose answers live in a table that no strategy makes retrievable
- one question where a chunk from a different paper (BERT) supplies a
  plausible-looking number for a question about the Transformer

### Reranking is where the value is

| k | dense | BM25 | hybrid | hybrid + rerank |
|---|---|---|---|---|
| 3 | 0.69 | 0.62 | 0.69 | **0.94** |
| 5 | 0.69 | 0.75 | 0.81 | 0.94 |
| 10 | 0.94 | 0.81 | 0.88 | **1.00** |

Two distinct failure modes fall out of this table. **Dense retrieval finds almost
everything but ranks it deep** — 0.69 at k=3 rising to 0.94 at k=10. **BM25 is
binary** — 0.75 at k=5, then nothing more. Reranking fixes ranking, which is why
it helps so much at k=3, the realistic prompt budget.

It also shows where RRF *hurts*: at k=10 it dilutes a strong dense result with
weaker lexical candidates (0.94 → 0.88).

### Markdown markup silently breaks lexical search

`_dk_` (markdown italics) and `dk` tokenize to different BM25 terms. Markup left
in the index made key technical terms invisible to lexical retrieval:

```
query: "why divide by square root of dk"

before cleaning:  2 results, top hit wrong (an optimizer section)
after cleaning:   5 results, correct chunk at #1, the answer chunk at #3
```

Dense retrieval was unaffected — a few extra symbols vanish when 1000 characters
are averaged into 384 numbers. The failure was specific to exact-match search.

### Faithfulness cannot be read without correctness

Adding reranking to the heading-based index **lowered** faithfulness (0.84 → 0.77)
while **raising** correctness (0.62 → 0.69).

Without reranking the model more often received irrelevant context and honestly
refused — and a refusal scores 1.0 on faithfulness. With better context it
answered, and occasionally added an unsupported hedge.

**High faithfulness can simply mean the system stays silent more often.**

---

## LLM-as-judge, and why it needed calibration

Ragas was dropped — its dependency chain conflicts with vLLM over `openai`
versions. The judge here is ~60 lines: one prompt decomposes an answer into
factual claims and checks each against the context, another compares the answer
to a reference.

The first version of the judge **systematically penalised short answers**. It
split single sentences into fragments that could not be verified individually:

```
"The positional encoding uses sine and cosine functions of different
 frequencies [1]"  →  faithfulness 0.33
```

That answer is a direct quote from the context. The score was an artifact of the
measurement, not a property of the answer.

Fixing the judge prompt — *"do not split a single claim into fragments; a short
answer may contain exactly one claim"* — barely moved the mean (0.79 → 0.77) but
made the distribution honest: scores collapsed to 0 on the cases that were
genuinely unsupported, instead of earning 0.75 from trivially-true fragments.

**An LLM judge measures its own behaviour until you check individual cases by
hand.**

---

## Three failure types, and what each one means

The two metrics together form a diagnostic:

| faithfulness | correctness | diagnosis | root cause |
|---|---|---|---|
| 0 | 0 | fabricated | answer absent from context → **retrieval** |
| 1 | 0 | wrong fact, honestly cited | similar-but-different chunk → **retrieval** |
| 0.5 | 0 | hedged | unsupported caveats added → generation |

Every persistent failure except hedging originates in retrieval. Generation is
not the bottleneck.

### The unfixable one: tables

Two questions fail across every configuration. Both have answers in Table 3 of
the Transformer paper — a chunk that contains **both the caption and the table**:

```
Table 3: Variations on the Transformer architecture. Unlisted values are
identical to those of the base model...
| base | 6 | 512  | 2048 | 8  | ... | 65  |
| big  | 6 | 1024 | 4096 | 16 | ... | 213 |
```

The caption carries the right words. The table carries the number. It still
doesn't retrieve:

- **dense** — 400 characters of meaningful caption drown in 900 characters of
  digits and pipes when averaged into a single vector
- **BM25** — the query says `parameters`, the table header says `params`, and
  stemming does not bridge them

This is not a parsing or chunking defect. It is a signal-to-noise problem inside
a chunk with mixed content, and the fix is contextual enrichment — generating a
prose description of the table at ingest time — which is left as future work.

---

## Running it

```bash
pip install qdrant-client sentence-transformers fastembed pyyaml httpx
# torch separately, matched to your hardware

python -m src.ingest                                      # PDF → markdown
python -m src.index_hybrid --strategy headings --collection rag_main
python -m src.search_hybrid "why are the dot products scaled"
python -m src.generate "how many attention heads does the base Transformer use"
```

Evaluation:

```bash
python -m src.eval_retrieval --collection rag_main --k 3   # recall@k, MRR
python -m src.run_eval rag_main                            # generate answers
python -m src.judge results_rag_main.json                  # faithfulness, correctness
```

Generation requires an OpenAI-compatible endpoint on `localhost:8000` — vLLM in
this setup, Ollama works too with a one-line change.

---

## Layout

```
src/
  ingest.py          PDF → markdown
  chunking.py        three strategies, markdown cleaning, noise filtering
  index_hybrid.py    dense + sparse vectors into Qdrant
  search_hybrid.py   dense / BM25 / RRF fusion
  rerank.py          cross-encoder second stage
  generate.py        RAG prompt, citation, refusal
  eval_retrieval.py  recall@k, MRR against labelled chunks
  run_eval.py        batch generation over the dataset
  judge.py           LLM-as-judge: faithfulness, correctness
datasets/
  retrieval.yaml     16 hand-labelled questions with relevant chunks
                     and reference answers
```

---

## Known limitations

- **Retrieval ground truth is tied to chunk indices**, which shift when the
  chunking strategy changes. Comparing strategies by `recall@k` is therefore
  invalid across indexes; the fix is to anchor labels to text (`must_contain`)
  rather than position. End-to-end metrics are unaffected.
- **16 questions is a small dataset.** Differences under ~0.1 are one question
  and should not be read as signal.
- **The judge is a 7B model** and is not calibrated against human judgement.
  Relative comparisons hold; absolute values should not be quoted.
- Formulas do not survive PDF parsing in a searchable form.

---

## Related

[llm-playground](https://github.com/HoLokostikk/llm-playground) — inference
benchmarks behind the serving choices here: engine comparison, prefill/decode
characteristics, KV-cache memory planning.
