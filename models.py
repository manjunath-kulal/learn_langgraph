import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found")

genai.configure(api_key=api_key)

try:
    models = genai.list_models()
    print("Gemini API key is VALID. Available models:\n")
    for m in models:
        print(m.name)
except Exception as e:
    print("Error accessing Gemini API:")
    print(e)