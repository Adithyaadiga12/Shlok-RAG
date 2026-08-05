"""
build_index.py — turn data/rag/verses.jsonl into a searchable FAISS index.

For each verse we build one short text (transliteration + English translation when
available), embed it with a multilingual sentence-transformer, and store all the
vectors in a FAISS index. A parallel meta.json keeps the display fields so a search
hit can be shown back to the user.

Run once (slow — it embeds every verse):
    python rag/build_index.py                 # full corpus
    python rag/build_index.py --limit 5000     # quick test on a subset

Outputs (into rag/index/):
    shlok.faiss   the vector index
    meta.json     list of {id, source, category, chapter, verse, sanskrit, translation}
    config.json   which embedding model was used (so search uses the same one)
"""
import argparse
import json
import os

import numpy as np

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# paths can be overridden by env vars (handy on Kaggle, where the data lives
# under /kaggle/input/... and output must go to /kaggle/working)
VERSES   = os.environ.get("VERSES_PATH", os.path.join(ROOT, "data", "rag", "verses.jsonl"))
OUT_DIR  = os.environ.get("INDEX_OUT",   os.path.join(os.path.dirname(os.path.abspath(__file__)), "index"))

# multilingual model — handles Devanagari + English, and aligns them cross-lingually
# so an English query can match a Sanskrit verse. 384-dim, small + fast.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def best_translation(record):
    """Return one English translation string, or '' if none."""
    tr = record.get("translations") or {}
    if isinstance(tr, dict):
        for v in tr.values():
            if isinstance(v, str) and len(v.strip()) > 10:
                return v.strip()
    return ""


def embed_text(record):
    """The text we actually embed for a verse: transliteration + translation."""
    parts = []
    if record.get("transliteration"):
        parts.append(record["transliteration"].replace("\n", " ").strip())
    tr = best_translation(record)
    if tr:
        parts.append(tr)
    if not parts and record.get("sanskrit"):          # fallback
        parts.append(record["sanskrit"].replace("\n", " ").strip())
    return " . ".join(parts)


def meta_of(record):
    return {
        "id":          record.get("id"),
        "source":      record.get("source"),
        "category":    record.get("category"),
        "chapter":     record.get("chapter"),
        "verse":       record.get("verse"),
        "sanskrit":    record.get("sanskrit", "").strip(),
        "translation": best_translation(record),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="embed only first N verses (0 = all)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    # imported here so --help works without the heavy deps installed
    import faiss
    from sentence_transformers import SentenceTransformer

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. read verses + build the text to embed ----
    texts, metas = [], []
    with open(VERSES, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            r = json.loads(line)
            t = embed_text(r)
            if not t:
                continue
            texts.append(t)
            metas.append(meta_of(r))
    print(f"prepared {len(texts):,} verses to embed")

    # ---- 2. embed in batches ----
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    print(f"model {args.model} (dim {dim}) — embedding...")

    embs = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,          # so inner-product == cosine similarity
        convert_to_numpy=True,
    ).astype(np.float32)

    # ---- 3. build + save FAISS index ----
    index = faiss.IndexFlatIP(dim)          # exact cosine search (fine up to ~1M verses)
    index.add(embs)
    faiss.write_index(index, os.path.join(OUT_DIR, "shlok.faiss"))

    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "dim": dim, "count": len(metas)}, f)

    print(f"done. index={index.ntotal:,} vectors  ->  {OUT_DIR}")


if __name__ == "__main__":
    main()
