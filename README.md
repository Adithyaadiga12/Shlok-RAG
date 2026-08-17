# ShlokGPT-RAG

Semantic search + question-answering over classical Sanskrit texts — the
**Rāmāyaṇa, Mahābhārata, and Bhagavad Gītā** (~93,700 English-translated verses).
Ask a question in English → get a grounded answer with the exact verses it's built from.

Companion to [ShlokGPT](https://github.com/Adithyaadiga12/ShlokGPT) (a Sanskrit GPT trained from scratch).

## What it does
- **`/search`** — semantic search: your question → the most relevant verses
- **`/ask`** — RAG: retrieves verses, then an LLM writes a grounded answer that **cites the verse IDs** and shows the original Sanskrit

## Pipeline
```
question
  → embed + FAISS search (93K verses → 20 candidates)     [bi-encoder]
  → cross-encoder reranker (re-sorts the 20)              [stage 2]
  → dedup + relevance gate (drop repeats / off-topic)
  → build grounded prompt → Gemini → answer + citations + sources
```

Two-stage retrieval (bi-encoder → cross-encoder reranker), near-duplicate
de-duplication, and a min-score gate that returns *"no relevant verses"* on
off-topic queries.

## Evaluation
Built a 75-question eval set (an LLM writes a question per verse, so the gold
verse is known) and measured retrieval:

| Metric | No reranker | + reranker |
|---|---|---|
| Recall@1 | 0.11 | **0.27** |
| MRR@5 | 0.19 | **0.30** |
| Recall@5 | 0.31 | 0.35 |

Manual relevance judgment: **context relevance ≈ 0.68**, with a relevant verse
in the top-5 for **~92%** of queries. (Exact-ID recall is capped by the
Mahābhārata paralleling the Gītā — many "misses" retrieve the equivalent verse
under a different ID.)

## Run locally
```bash
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key" > .env      # free key from aistudio.google.com
python api.py
```
Open http://127.0.0.1:8000/docs

Requires the FAISS index in `index/` (`shlok.faiss`, `meta.json`, `config.json`).
Rebuild it with `python build_index.py` (needs `verses.jsonl` from the dataset;
set `VERSES_PATH` to point at it).

## Files
- `api.py` — FastAPI service (`/search`, `/ask`)
- `ask.py` — prompt building + Gemini call
- `build_index.py` — build the FAISS index from translated verses
- `measure.py` — retrieval eval (Recall@k, MRR)
- `measure_generation.py` — generation eval (RAG triad via LLM judge)
- `eval/gita_eval.json` — the evaluation set
