from ollama import chat

def generate_answer(question, context):

    prompt = f"""
You are an AI assistant for AIMS Senegal.

Answer ONLY using the context below.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = chat(
        model="phi3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]