from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ollama import chat

# =====================
# Charger la base vectorielle
# =====================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="vectordb",
    embedding_function=embeddings
)

# =====================
# Question utilisateur
# =====================

question = input("Question : ")

# =====================
# Recherche RAG
# =====================

docs = db.similarity_search(
    question,
    k=3
)

context = "\n\n".join(
    doc.page_content for doc in docs
)

# =====================
# Prompt
# =====================

prompt = f"""
You are an AI teaching assistant.

Use ONLY the context below to answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# =====================
# Phi-3
# =====================

response = chat(
    model="phi3",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\n")
print("=" * 80)
print("REPONSE")
print("=" * 80)
print(response.message.content)