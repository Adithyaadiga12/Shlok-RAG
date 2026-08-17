"""
measure.py — Phase 3: measure retrieval quality using the eval set.

For each {question, gold_verse_id} in eval/gita_eval.json, run the question
through search and check whether the gold verse is in the top-k results.
Reports Recall@1, Recall@k, and MRR.

Run (in a separate terminal from build_eval.py):
    python rag/measure.py
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api import search          # reuse your exact /search logic

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "eval", "gita_eval.json")


def measure(k=5):
    data = json.load(open(EVAL, encoding="utf-8"))
    hits_at_1 = hits_at_k = 0
    mrr_total = 0.0

    for row in data:
        gold = row["gold_verse_id"]
        results = search(row["question"], k)["results"]   # run the search
        ids = [r["id"] for r in results]                  # the returned verse IDs

        if gold in ids:
            rank = ids.index(gold) + 1        # position of gold verse (1-based)
            hits_at_k += 1
            mrr_total += 1.0 / rank
            if rank == 1:
                hits_at_1 += 1

    n = len(data)
    print(f"\neval questions : {n}")
    print(f"Recall@1       : {hits_at_1/n:.3f}  ({hits_at_1}/{n})")
    print(f"Recall@{k}       : {hits_at_k/n:.3f}  ({hits_at_k}/{n})")
    print(f"MRR@{k}          : {mrr_total/n:.3f}")


if __name__ == "__main__":
    measure(k=5)
