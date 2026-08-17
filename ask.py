import os 
from dotenv import load_dotenv
from google import genai

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
  
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) 
    
import concurrent.futures

# persistent executor — do NOT use a `with` block (its __exit__ waits for the
# thread, which would cancel out the timeout and hang the app)
_EXEC = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def gemini_call(prompt, model="gemini-flash-lite-latest"):   # stable alias, highest free quota
    """Call Gemini with a HARD 15s cap so a throttled/retrying request never hangs
    the app. On timeout/error we return a friendly note; the retrieved verses are
    still shown by the caller."""
    fut = _EXEC.submit(lambda: _client.models.generate_content(model=model, contents=prompt).text)
    try:
        return fut.result(timeout=15)
    except Exception:
        return ("(The answer service is busy right now — see the retrieved verses below. "
                "Please try again in a moment.)")
