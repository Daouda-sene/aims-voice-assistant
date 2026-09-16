import os
from groq import Groq


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
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
