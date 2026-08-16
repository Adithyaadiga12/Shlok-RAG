"""
measure_generation.py — Phase 4: measure GENERATION quality (the RAG triad).

For a small sample of eval questions, run the full RAG pipeline (retrieve ->
answer), then use Gemini as a JUDGE to score three things (0.0-1.0):

  1. context_relevance  — are the retrieved verses relevant to the question?
  2. groundedness       — does the answer only use the retrieved verses?
  3. answer_relevance   — does the answer actually address the question?

Free-tier friendly: uses gemini-2.5-flash-lite (its own daily quota) and scores
all 3 metrics in ONE judge call, so each question costs just 2 Gemini requests.

Run:  python rag/measure_generation.py
"""
import json
import os
import re
import sys
import time
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api import search
from ask import build_prompt, gemini_call

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL  = os.path.join(ROOT, "eval", "gita_eval.json")
MODEL = "gemini-2.5-flash-lite"          # separate free-tier quota from 2.5-flash


def verses_text(hits):
    return "\n".join(f"- [{h['id']}] {h['translation']}" for h in hits)


def _num(reply, key):
    """Pull `key=0.8` style number out of the judge reply."""
    m = re.search(rf"{key}\s*[=:]\s*([01](?:\.\d+)?)", reply, re.I)
    return float(m.group(1)) if m else 0.0


def judge_all(question, ctx, answer):
    """One call scores all three triad metrics."""
    prompt = (
        f"Question: {question}\n\nRetrieved verses:\n{ctx}\n\nAnswer: {answer}\n\n"
        "Score each of these from 0.0 to 1.0:\n"
        "context_relevance = are the verses relevant to the question?\n"
        "groundedness = is the answer supported ONLY by the verses (no made-up info)?\n"
        "answer_relevance = does the answer address the question?\n\n"
        "Reply on ONE line exactly like:\n"
        "context_relevance=0.9 groundedness=1.0 answer_relevance=0.8"
    )
    for attempt in range(4):
        try:
            reply = gemini_call(prompt, model=MODEL)
            return (_num(reply, "context_relevance"),
                    _num(reply, "groundedness"),
                    _num(reply, "answer_relevance"))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(30)


def main(n=8, sleep=5.0):
    data = json.load(open(EVAL, encoding="utf-8"))
    random.seed(42)
    sample = random.sample(data, min(n, len(data)))

    totals = [0.0, 0.0, 0.0]
    done = 0
    for i, row in enumerate(sample):
        q = row["question"]
        hits = search(q, 5)["results"]                          # retrieve (no LLM call)
        answer = gemini_call(build_prompt(q, hits), model=MODEL)  # 1 call: generate
        cr, gr, ar = judge_all(q, verses_text(hits), answer)      # 1 call: judge all 3

        totals[0] += cr; totals[1] += gr; totals[2] += ar
        done += 1
        print(f"[{i+1}/{len(sample)}] ctx={cr:.2f} ground={gr:.2f} ans={ar:.2f}  | {q[:45]}")
        time.sleep(sleep)

    print(f"\n--- RAG triad over {done} questions ---")
    print(f"Context Relevance : {totals[0]/done:.3f}")
    print(f"Groundedness      : {totals[1]/done:.3f}")
    print(f"Answer Relevance  : {totals[2]/done:.3f}")


if __name__ == "__main__":
    main()
