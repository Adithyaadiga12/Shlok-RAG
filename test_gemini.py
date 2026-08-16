from dotenv import load_dotenv
import os
load_dotenv()# Load environment variables from .env file
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents = "Say Hello in Sanskrit"
)

print(response.text)