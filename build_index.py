"""
build_index.py — build the searchable FAISS index for the RAG app.

The app serves ENGLISH questions, so we only index verses that HAVE an English
translation (~93K: the full Ramayana + Mahabharata + Gita). Reasons:
  - an English query matches an English translation directly — fast, accurate,
    and NO per-query translation step (so no added latency)
  - an English user can actually read the result

We EMBED the English translation (for matching) but STORE the Sanskrit verse,
transliteration, and source too (for display). The untranslated verses aren't
lost from the project — they were used to train the GPT model; they just can't
serve an English-facing search.

Run:
    python rag/build_index.py                # full (~93K, a few min on CPU)
    python rag/build_index.py --limit 3000   # quick test

Outputs into rag/index/: shlok.faiss, meta.json, config.json
"""
import argparse
import json
import os

import numpy as np

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSES   = os.environ.get("VERSES_PATH", os.path.join(ROOT, "data", "rag", "verses.jsonl"))
OUT_DIR  = os.environ.get("INDEX_OUT",   os.path.join(os.path.dirname(os.path.abspath(__file__)), "index"))

# multilingual model — good enough for English translation search, small + fast
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def best_translation(record):
    """Return one English translation string, or '' if none."""
    tr = record.get("translations") or {}
    if isinstance(tr, dict):
        for v in tr.values():
            if isinstance(v, str) and len(v.strip()) > 10:
                return v.strip()
    return ""


def meta_of(record, translation):
    """The display card kept for each verse (shown in search results)."""
    return {
        "id":              record.get("id"),
        "source":          record.get("source"),
        "category":        record.get("category"),
        "chapter":         record.get("chapter"),
        "verse":           record.get("verse"),
        "sanskrit":        record.get("sanskrit", "").strip(),
        "transliteration": (record.get("transliteration") or "").strip(),
        "translation":     translation,
    }


def load_reference(rag_dir):
    """Also index curated entities (who is X) + glossary (what is X).
    Returns list of (text_to_embed, meta_card)."""
    out = []
    ent = os.path.join(rag_dir, "entities.json")
    glo = os.path.join(rag_dir, "glossary.json")

    if os.path.exists(ent):
        for e in json.load(open(ent, encoding="utf-8")):
            aliases = ", ".join(e.get("aliases", []))
            head = f"{e['name']} ({aliases})" if aliases else e["name"]
            out.append((f"{head}: {e['description']}", {
                "id": e["id"], "source": "Entity (" + ", ".join(e.get("sources", [])) + ")",
                "category": "Entity", "chapter": None, "verse": None,
                "sanskrit": e.get("sanskrit", ""), "transliteration": "",
                "translation": e["description"],
            }))

    if os.path.exists(glo):
        for g in json.load(open(glo, encoding="utf-8")):
            out.append((f"{g['term']}: {g['definition']}", {
                "id": g["id"], "source": "Glossary",
                "category": "Glossary", "chapter": None, "verse": None,
                "sanskrit": g.get("sanskrit", ""), "transliteration": "",
                "translation": g["definition"],
            }))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="index only first N translated verses (0 = all)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    import faiss
    from sentence_transformers import SentenceTransformer

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. read verses, keep only those WITH an English translation ----
    texts, metas = [], []
    kept = skipped = 0
    with open(VERSES, encoding="utf-8") as f:
        for line in f:
            if args.limit and kept >= args.limit:
                break
            r = json.loads(line)
            tr = best_translation(r)
            if not tr:                          # no English -> skip (English search can't use it)
                skipped += 1
                continue
            texts.append(tr)                    # EMBED the English translation
            metas.append(meta_of(r, tr))        # STORE Sanskrit + translation for display
            kept += 1
    print(f"indexed {kept:,} translated verses (skipped {skipped:,} without translation)")

    # ---- also index curated entities + glossary (always, for "who/what is X") ----
    ref = load_reference(os.path.dirname(VERSES))
    for t, m in ref:
        texts.append(t)
        metas.append(m)
    print(f"+ {len(ref)} reference entries (entities + glossary)")

    # ---- 2. embed the translations ----
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    print(f"model {args.model} (dim {dim}) — embedding...")
    embs = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    # ---- 3. build + save the FAISS index ----
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, os.path.join(OUT_DIR, "shlok.faiss"))
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "dim": dim, "count": len(metas)}, f)

    print(f"done. index={index.ntotal:,} vectors  ->  {OUT_DIR}")


if __name__ == "__main__":
    main()
