import os 
from dotenv import load_dotenv
from google import genai

load_dotenv()  # Load environment variables from .env file



def build_prompt(question,verses):
    #instructions (the "rules" for the LLM
    instructions = (
        "You are a knowledgeable Sanskrit scholar.\n"
        "Answer the question using ONLY the verses provided below.\n"
        "If the verses don't contain the answer, say so honestly — do not make up verses.\n"
        "Cite the verse IDs (like BG3.35) you used in your answer.\n"
    )
    
    context_lines =[]
    for verse in verses:
        line = f"-[{verse['id']}]  {verse['translation']}"
        if not verse['translation']:
            line = f"-[{verse['id']}]  {verse['sanskrit']}"
        context_lines.append(line)
    
    context = "Verse:\n" + "\n".join(context_lines)
    
    question_block = f"Question:\n{question}"
    
    return instructions + "\n" + context + "\n" + question_block
  
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) 
    
def gemini_call(prompt,model = "gemini-2.5-flash"):
    response = _client.models.generate_content(model = model,contents = prompt)
    return response.text
