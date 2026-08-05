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

import faiss
import numpy as np
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer

INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index")

# ---- load index + metadata + the SAME model used to build it ----
with open(os.path.join(INDEX_DIR, "config.json")) as f:
    CFG = json.load(f)
with open(os.path.join(INDEX_DIR, "meta.json"), encoding="utf-8") as f:
    META = json.load(f)

INDEX = faiss.read_index(os.path.join(INDEX_DIR, "shlok.faiss"))
MODEL = SentenceTransformer(CFG["model"])

app = FastAPI(title="ShlokGPT RAG", description="Semantic search over Sanskrit verses")


@app.get("/")
def health():
    return {"status": "ok", "verses_indexed": INDEX.ntotal, "model": CFG["model"]}


@app.get("/search")
def search(q: str = Query(..., description="natural-language or Sanskrit query"),
           k: int = Query(5, ge=1, le=50)):
    """Return the k verses most semantically similar to the query."""
    vec = MODEL.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    scores, idxs = INDEX.search(vec, k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i < 0:
            continue
        m = dict(META[i])
        m["score"] = round(float(score), 4)
        results.append(m)
    return {"query": q, "results": results}


if __name__ == "__main__":
    # lets you start the server with:  python rag/api.py
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
