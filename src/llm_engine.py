import os
from groq import Groq


api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not configured. "
        "Please add it in Streamlit Cloud → Settings → Secrets."
    )


client = Groq(
    api_key=api_key
)


def generate_answer(question, context):

    prompt = f"""
You are an AI assistant for AIMS Senegal.

Answer ONLY using the context below.

If the answer is not contained in the context,
say that you do not have enough information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content
