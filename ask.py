import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # Load environment variables from .env file



def build_prompt(question,verses):
    #instructions (the "rules" for the LLM)
    instructions = (
        "You are a knowledgeable Sanskrit scholar.\n"
        "Use the retrieved verses below as your PRIMARY source, and cite the verse IDs "
        "you draw from, inline, like [BG3.35].\n"
        "You MAY add well-established general knowledge to give a complete, helpful answer "
        "— but never invent verse text or attribute claims to verses that don't support them.\n"
        "If the verses don't directly cover the question, still answer helpfully from your "
        "knowledge, and note what the provided verses do or don't address.\n"
        "Keep the answer concise and clear.\n"
    )
    

    context_lines = []
    for verse in verses:
        text = verse['translation'] or verse['sanskrit']      # fall back to Sanskrit if no translation
        context_lines.append(f"- [{verse['id']}] {text}")

    context = "Verses:\n" + "\n".join(context_lines)
    question_block = f"Question:\n{question}\n\nAnswer:"

    return instructions + "\n" + context + "\n\n" + question_block
  
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

import concurrent.futures

# persistent executor — do NOT use a `with` block (its __exit__ waits for the
# thread, which would cancel out the timeout and hang the app)
_EXEC = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def gemini_call(prompt, model="openai/gpt-oss-120b"):   # Groq: fast (LPU) + strong open model
    """Call the LLM (Groq) with a HARD 15s cap so a slow/throttled request never hangs
    the app. On timeout/error we return a friendly note; the retrieved verses are
    still shown by the caller."""
    def _run():
        resp = _client.chat.completions.create(
            model=model,
            reasoning_effort="low",     # skip heavy reasoning -> much faster for grounded summarizing
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    fut = _EXEC.submit(_run)
    try:
        return fut.result(timeout=15)
    except Exception:
        return ("(The answer service is busy right now — see the retrieved verses below. "
                "Please try again in a moment.)")
