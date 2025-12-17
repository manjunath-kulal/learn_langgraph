import os
from dotenv import load_dotenv
from google import genai

# Load env vars
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found")

# Create client
client = genai.Client(api_key=api_key)

def test_prompt():
    response = client.models.generate_content(
        model="models/gemini-2.5-flash-lite",
        contents="Explain what FAISS is in one sentence."
    )

    print("\nGemini response:\n")
    print(response.text)

if __name__ == "__main__":
    test_prompt()