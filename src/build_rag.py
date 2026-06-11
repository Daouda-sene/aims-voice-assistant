from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =========================
# Configuration
# =========================

PDF_FOLDER = Path("data/pdfs")
VECTOR_DB_DIR = "vectordb"

# =========================
# Vérification du dossier
# =========================

if not PDF_FOLDER.exists():
    raise FileNotFoundError(
        f"Le dossier {PDF_FOLDER} n'existe pas."
    )

pdf_files = list(PDF_FOLDER.glob("*.pdf"))

if len(pdf_files) == 0:
    raise ValueError(
        f"Aucun PDF trouvé dans {PDF_FOLDER}"
    )

print(f"\n{len(pdf_files)} PDF(s) trouvé(s).\n")

# =========================
# Chargement des PDF
# =========================

documents = []

for pdf_file in pdf_files:
    try:
        print(f"Chargement : {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        documents.extend(docs)

    except Exception as e:
        print(f"Erreur avec {pdf_file.name} : {e}")

print(f"\nNombre total de pages chargées : {len(documents)}")

if len(documents) == 0:
    raise ValueError(
        "Aucune page n'a été chargée."
    )

# =========================
# Découpage en chunks
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"Nombre de chunks créés : {len(chunks)}")

# =========================
# Embeddings
# =========================

print("\nChargement du modèle d'embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# Création de la base vectorielle
# =========================

print("\nCréation de la base vectorielle...")

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=VECTOR_DB_DIR
)

print("\n✅ Base vectorielle créée avec succès !")
print(f"📁 Répertoire : {VECTOR_DB_DIR}")
print(f"📄 Pages : {len(documents)}")
print(f"🧩 Chunks : {len(chunks)}")