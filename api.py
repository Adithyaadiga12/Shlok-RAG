"""
api.py — FastAPI service over the FAISS index built by build_index.py.

Endpoints:
    GET  /            health + how many verses are indexed
    GET  /search?q=...&k=5   semantic search -> top-k verses with sources

Run:
    uvicorn rag.api:app --reload      (from the project root)
then open http://127.0.0.1:8000/docs for an interactive UI.
"""
import json
import os
from difflib import SequenceMatcher

import faiss
import numpy as np
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer, CrossEncoder
from ask import gemini_call,build_prompt

INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index")

# ---- load index + metadata + the SAME model used to build it ----
with open(os.path.join(INDEX_DIR, "config.json")) as f:
    CFG = json.load(f)
with open(os.path.join(INDEX_DIR, "meta.json"), encoding="utf-8") as f:
    META = json.load(f)

INDEX = faiss.read_index(os.path.join(INDEX_DIR, "shlok.faiss"))
MODEL = SentenceTransformer(CFG["model"])

# ---- reranker: a local cross-encoder (no API, no Gemini). Stage 2 of retrieval. ----
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# below this stage-1 cosine, we treat the query as having NO relevant verse
MIN_SCORE = 0.2


def dedup(results, threshold=0.85):
    """Drop near-duplicate verses (e.g. a Gita verse and its Mahabharata twin).
    Two verses are 'the same' if their Sanskrit text is >85% similar."""
    kept = []
    for r in results:
        s = "".join(r.get("sanskrit", "").split())          # strip whitespace
        if s and any(SequenceMatcher(None, s, "".join(k.get("sanskrit", "").split())).ratio() > threshold
                     for k in kept):
            continue                                          # too similar to one we kept -> skip
        kept.append(r)
    return kept

app = FastAPI(title="ShlokGPT RAG", description="Semantic search over Sanskrit verses")


@app.get("/")
def health():
    return {"status": "ok", "verses_indexed": INDEX.ntotal, "model": CFG["model"]}


@app.get("/search")
def search(q: str = Query(..., description="natural-language or Sanskrit query"),
           k: int = Query(5, ge=1, le=50),
           rerank: bool = Query(True, description="use the cross-encoder reranker (stage 2)")):
    """Two-stage retrieval: FAISS grabs candidates, then the reranker re-sorts them."""
    # --- stage 1: fast FAISS search grabs a wider candidate pool ---
    n_candidates = max(k * 4, 20) if rerank else k
    vec = MODEL.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    scores, idxs = INDEX.search(vec, n_candidates)
    cands = []
    for score, i in zip(scores[0], idxs[0]):
        if i < 0:
            continue
        m = dict(META[i])
        m["score"] = round(float(score), 4)          # stage-1 (vector) score
        cands.append(m)

    # --- relevance gate: if even the best candidate is weak, return nothing ---
    if not cands or max(c["score"] for c in cands) < MIN_SCORE:
        return {"query": q, "results": []}

    # --- stage 2: reranker re-reads (question, verse) pairs and re-sorts ---
    if rerank and cands:
        pairs = [(q, c["translation"]) for c in cands]
        rerank_scores = RERANKER.predict(pairs)
        for c, rs in zip(cands, rerank_scores):
            c["rerank_score"] = round(float(rs), 4)
        cands.sort(key=lambda c: c["rerank_score"], reverse=True)

    cands = dedup(cands)                 # remove near-duplicate verses
    return {"query": q, "results": cands[:k]}



@app.get("/ask")
def ask( q : str =Query(..., description="your question in natural language or Sanskrit") ,
        k : int=Query(5, ge=1, le=50)):
    """Return the answer to the question using the k most relevant verses."""
    hits = search(q,k)["results"]
    if not hits:                          # nothing relevant -> don't call Gemini (saves quota)
        return {"Query": q, "Answer": "I couldn't find any relevant verses for that question.", "Sources": []}
    prompt = build_prompt(q,hits)
    answer = gemini_call(prompt)
    return {"Query": q, "Answer": answer, "Sources": hits}


if __name__ == "__main__":
    # lets you start the server with:  python rag/api.py
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


