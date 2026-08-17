import os 
from dotenv import load_dotenv
from google import genai

load_dotenv()  # Load environment variables from .env file



def build_prompt(question,verses):
    #instructions (the "rules" for the LLM)
    instructions = (
        "You are a knowledgeable Sanskrit scholar.\n"
        "Answer the question using ONLY the verses provided below — do not add outside "
        "knowledge or invent verses.\n"
        "If the verses only partially address the question, answer what they support and "
        "note what they do not cover.\n"
        "If none of the verses are relevant, say so honestly.\n"
        "Cite the verse IDs you used inline, like [BG3.35].\n"
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
    
def gemini_call(prompt,model = "gemini-2.5-flash"):
    response = _client.models.generate_content(model = model,contents = prompt)
    return response.text
