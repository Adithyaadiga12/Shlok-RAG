"""
build_eval.py — build an evaluation set for measuring retrieval quality.

Picks 150 Gita verses, asks Gemini to write a question for each, and saves
{question, gold_verse_id} pairs. Resumable — rerun to continue if it stops.
"""
import json
import os
import sys
import time
import random

HERE   = os.path.dirname(os.path.abspath(__file__))
# verses.jsonl comes from the dataset repo; override with VERSES_PATH env var
VERSES = os.environ.get("VERSES_PATH",
                        os.path.join(os.path.dirname(HERE), "data", "rag", "verses.jsonl"))
OUT    = os.path.join(HERE, "eval", "gita_eval.json")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # so we can import ask.py
from ask import gemini_call


def best_translation(record):
    """Return one English translation string, or '' if none."""
    tr = record.get("translations") or {}
    if isinstance(tr, dict):
        for v in tr.values():
            if isinstance(v, str) and len(v.strip()) > 10:
                return v.strip()
    return ""


def load_gita_verses():
    """Read verses.jsonl, keep only Gita verses with a translation."""
    verses = []
    with open(VERSES, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("category") != "Gita":
                continue
            tr = best_translation(r)
            if not tr:
                continue
            verses.append({"id": r["id"], "translation": tr})
    return verses


def make_question(verse):
    """Ask Gemini to write a natural question that this verse answers."""
    prompt = (
        "Here is an English translation of a Sanskrit verse:\n\n"
        f'"{verse["translation"]}"\n\n'
        "Write ONE natural question that a person might ask, where THIS verse "
        "would be a good answer. The question should be about the meaning or "
        "teaching, not mention verse numbers. Reply with only the question."
    )
    # retry on rate limits: wait and try again a few times before giving up
    for attempt in range(4):
        try:
            return gemini_call(prompt).strip()
        except Exception as e:
            if attempt == 3:
                raise
            print(f"    rate limited ({type(e).__name__}); waiting 30s...")
            time.sleep(30)


def build_eval_set(n=150, sleep=7.0):        # 7s => under 10 requests/min
    verses = load_gita_verses()
    random.seed(42)
    sample = random.sample(verses, min(n, len(verses)))

    done = {}
    if os.path.exists(OUT):
        done = {row["gold_verse_id"]: row for row in json.load(open(OUT, encoding="utf-8"))}
        print(f"resuming — {len(done)} already done")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = list(done.values())

    for i, v in enumerate(sample):
        if v["id"] in done:
            continue
        try:
            q = make_question(v)
            results.append({"question": q, "gold_verse_id": v["id"]})
            print(f"[{i+1}/{len(sample)}] {v['id']}: {q[:70]}")
        except Exception as e:
            print(f"[{i+1}/{len(sample)}] {v['id']} FAILED: {e}")   # full error
            break

        json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        time.sleep(sleep)

    print(f"\nsaved {len(results)} question→verse pairs to {OUT}")

if __name__ == "__main__":
    build_eval_set()