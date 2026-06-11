from faster_whisper import WhisperModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ollama import chat

# =====================
# AUDIO
# =====================

audio_file = "data/audio/question.wav"  # adaptez si besoin

print("Transcription de l'audio...")

whisper = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)

segments, info = whisper.transcribe(
    audio_file,
    beam_size=5
)

question = ""

for segment in segments:
    question += segment.text + " "

print("\nQUESTION :")
print(question)

# =====================
# RAG
# =====================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="vectordb",
    embedding_function=embeddings
)

docs = db.similarity_search(
    question,
    k=3
)

context = "\n\n".join(
    doc.page_content for doc in docs
)

# =====================
# LLM
# =====================

prompt = f"""
You are an AI teaching assistant.

Use only the context below.

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

print("\n" + "=" * 80)
print("REPONSE")
print("=" * 80)

print(response.message.content)