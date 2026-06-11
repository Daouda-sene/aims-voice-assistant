from faster_whisper import WhisperModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ======================
# AUDIO
# ======================

audio_file = "data/audio/question.wav"

print("Chargement de Whisper...")

model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe(
    audio_file,
    beam_size=5
)

question = ""

for segment in segments:
    question += segment.text + " "

print("\nQUESTION TRANSCRITE :")
print(question)

# ======================
# RAG
# ======================

print("\nChargement de la base vectorielle...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="vectordb",
    embedding_function=embeddings
)

results = db.similarity_search(
    question,
    k=3
)

print("\nRESULTATS RAG\n")

for i, doc in enumerate(results, start=1):
    print("=" * 80)
    print(f"DOCUMENT {i}")
    print("=" * 80)
    print(doc.page_content[:1000])