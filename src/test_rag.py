from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="vectordb",
    embedding_function=embeddings
)

question = input("Question : ")

results = db.similarity_search(
    question,
    k=3
)

for i, doc in enumerate(results, start=1):
    print("\n" + "=" * 80)
    print(f"RESULTAT {i}")
    print("=" * 80)

    print(doc.page_content[:1500])