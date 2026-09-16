import os
from dotenv import load_dotenv
load_dotenv(override=True)
from google import genai

client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
model = os.environ.get('LLM_MODEL', 'gemini-3.6-flash')
context = """
--- KNOWLEDGE BASE ---
Status: DIRECT
[
  {
    "id": "pH",
    "question": "What is the ideal pH range for Palak?",
    "answer": "The ideal pH range is 5.8 to 6.5."
  }
]

User: What is the ideal pH range for Palak?
"""
try:
    res = client.models.generate_content(
        model=model,
        contents=context,
        config={'max_output_tokens': 500, 'temperature': 0.3}
    )
    print("Response:", repr(res.text))
    if res.candidates:
        print("Finish reason:", res.candidates[0].finish_reason)
except Exception as e:
    print("Error:", e)
