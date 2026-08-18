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
import re
from difflib import SequenceMatcher

import faiss
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer, CrossEncoder
from ask import gemini_call,build_prompt

HERE      = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(HERE, "index")
STATIC    = os.path.join(HERE, "static")

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
def home():
    """Serve the web UI."""
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/health")
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



def structured_answer(q):
    """Answer counting / chapter-listing queries directly from metadata
    (no embeddings, no Gemini). Returns (answer, sources) or None."""
    ql = q.lower()

    # "verses/shlokas from chapter/adhyaya N"  -> list that Gita chapter
    m = re.search(r"(?:chapter|adhyaya|adhyaaya|adhyay)\s*(\d+)", ql)
    if m:
        ch = m.group(1)
        hits = [dict(x) for x in META if x.get("category") == "Gita" and str(x.get("chapter")) == ch]
        hits.sort(key=lambda x: int(x.get("verse") or 0))
        if hits:
            return (f"Chapter {ch} of the Bhagavad Gita has {len(hits)} verses — here they are:", hits)
        return (f"I don't have verses indexed for chapter {ch}.", [])

    # "how many verses/shlokas [in the Gita / epics / total]"  -> count
    if re.search(r"how many (verses|shlokas?|slokas?|shloka)", ql):
        if any(w in ql for w in ("gita", "gītā", "bhagavad")):
            n = sum(1 for x in META if x.get("category") == "Gita")
            return (f"There are {n} Bhagavad Gita verses indexed.", [])
        if any(w in ql for w in ("epic", "mahabharata", "mahābhārata", "ramayana", "rāmāyaṇa")):
            n = sum(1 for x in META if x.get("category") == "Epic")
            return (f"There are {n:,} epic (Ramayana + Mahabharata) verses indexed.", [])
        n = sum(1 for x in META if x.get("category") in ("Gita", "Epic"))
        return (f"There are {n:,} verses indexed in total.", [])

    return None


@app.get("/ask")
def ask( q : str =Query(..., description="your question in natural language or Sanskrit") ,
        k : int=Query(5, ge=1, le=50)):
    """Return the answer to the question using the k most relevant verses."""
    st = structured_answer(q)             # counts / chapter listings -> no embeddings, no Gemini
    if st is not None:
        return {"Query": q, "Answer": st[0], "Sources": st[1]}
    hits = search(q,k)["results"]
    if not hits:                          # nothing relevant -> don't call Gemini (saves quota)
        return {"Query": q, "Answer": "I couldn't find any relevant verses for that question.", "Sources": []}
    prompt = build_prompt(q,hits)
    answer = gemini_call(prompt)
    return {"Query": q, "Answer": answer, "Sources": hits}


if __name__ == "__main__":
    # 0.0.0.0 + PORT env so it works on Hugging Face Spaces (port 7860);
    # locally it serves on http://localhost:7860
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


