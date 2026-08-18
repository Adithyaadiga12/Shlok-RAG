# ShlokRAG

**Ask questions about classical Sanskrit scripture and get grounded, cited answers.**
A retrieval-augmented (RAG) app over the **Rāmāyaṇa, Mahābhārata & Bhagavad Gītā**
(~93,700 English-translated verses) plus a curated knowledge base of characters and terms.

Companion to **[ShlokGPT](https://github.com/Adithyaadiga12/ShlokGPT)** — a Sanskrit GPT trained from scratch, which produced the corpus this app searches.

---

## What it can do

- 💬 **Grounded Q&A** — ask in English, get an answer built *only* from retrieved verses, with inline `[verse-ID]` citations and the original Devanagari shown for verification.
- 👤 **"Who is X?"** — a curated set of 30 characters (Arjuna, Bhishma, the Pāṇḍavas…) so entity questions work, not just thematic ones.
- 📖 **"What is X?"** — 40 glossary terms (dharma, karma, mokṣa…) for concept definitions.
- 🔢 **Structural queries** — "how many verses are in the Gītā?" (→ 719) or "give me chapter 2" — answered directly from metadata, no LLM needed.
- 🛡️ **Honest by design** — a relevance gate returns *"no relevant verses"* for off-topic questions instead of hallucinating; the LLM cites what it uses.
- 🖥️ **Clean web UI** + a JSON API (`/search`, `/ask`).

## How it works

```
question
  → embed query + FAISS search   (93K verses → 20 candidates)   [bi-encoder]
  → cross-encoder reranker        (re-sorts the 20 for precision) [stage 2]
  → dedup + relevance gate        (drop near-duplicates / off-topic)
  → grounded prompt → LLM (Groq)  → answer + citations + source verses
```

**Two-stage retrieval** (bi-encoder → cross-encoder reranker), near-duplicate
de-duplication, a minimum-score "no answer" gate, and a hard timeout so a slow
LLM call degrades gracefully instead of hanging.

## Evaluation

Built a 75-question eval set (an LLM writes a question per verse, so the gold
verse is known by construction) and measured retrieval:

| Metric | No reranker | + reranker |
|---|---|---|
| **Recall@1** | 0.11 | **0.27** |
| **MRR@5** | 0.19 | **0.30** |
| Recall@5 | 0.31 | 0.35 |

Manual relevance judgment: **context relevance ≈ 0.68**, with a relevant verse in
the top-5 for **~92%** of queries. *(Exact-ID recall is capped because the
Mahābhārata parallels the Gītā — many "misses" retrieve the equivalent teaching
under a different verse ID.)*

Also profiled latency and cut answer time **188s → ~1s** by diagnosing free-tier
rate-limit backoff (retrieval itself is ~0.5s) and switching the LLM.

## Tech

`Python` · `FastAPI` · `Sentence-Transformers` · `FAISS` · `Cross-Encoder reranker`
· `Groq (LLM)` · `Uvicorn` · containerized with a `Dockerfile`

## Run locally

```bash
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key" > .env      # free key from console.groq.com
python api.py
```
Then open **http://localhost:7860** (or `/docs` for the API).

Requires the FAISS index in `index/` (`shlok.faiss`, `meta.json`, `config.json`).
Rebuild it with `python build_index.py` (needs `verses.jsonl` from the dataset —
set `VERSES_PATH` to point at it).

## Future work

Retrieval and evaluation are solid but not maxed out. Concrete, doable next steps:

- **Hybrid search (BM25 + dense)** — combine sparse keyword matching with the current
  embedding search. Would help exact-term queries (proper nouns, rare Sanskrit terms)
  that embeddings alone can blur; complements the cross-encoder reranker already in place.
- **RAGAS for generation eval** — swap the hand-rolled RAG-triad judge
  (`measure_generation.py`) for the standard RAGAS library (faithfulness, answer
  relevancy, context precision/recall) for more rigorously validated metrics.
- **Broaden the eval set beyond the Gītā** — the 75-question eval set is Gītā-only;
  generating questions across Purāṇa/Kāvya/Dharmaśāstra categories (once translated)
  would give a fuller picture of retrieval quality.
- **Citation accuracy checking** — programmatically verify every `[verse-ID]` cited in
  an answer is actually present in the retrieved sources (catches fabricated citations;
  currently only checked manually).
- **Latency profiling (p50/p95/p99)** — instrument each pipeline stage
  (embed / FAISS search / rerank / LLM) with `time.perf_counter()` and report
  percentiles, not just averages, to characterize the slow tail.
- **Response streaming** — stream the LLM's answer token-by-token instead of waiting
  for the full response, for a faster perceived time-to-first-token.
- **Semantic cache** — cache answers for repeated/near-duplicate questions to cut
  latency and LLM usage on a public deployment.
- **Deploy** — the app is containerized (`Dockerfile`) and ready for a host with a free
  compute tier (e.g. Google Cloud Run's scale-to-zero free tier).

## Project files

| File | Purpose |
|---|---|
| `api.py` | FastAPI service — `/search`, `/ask`, structural queries, serves the UI |
| `ask.py` | prompt building + LLM (Groq) call |
| `build_index.py` | build the FAISS index from verses + entities + glossary |
| `static/index.html` | the web UI |
| `measure.py` | retrieval eval (Recall@k, MRR) |
| `measure_generation.py` | generation eval (RAG triad via LLM judge) |
| `eval/gita_eval.json` | the evaluation set |

---

Built by **Adithya Adiga** · [adithyaadiga21@gmail.com](mailto:adithyaadiga21@gmail.com) · [github.com/Adithyaadiga12](https://github.com/Adithyaadiga12)
