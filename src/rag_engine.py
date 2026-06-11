from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Chargement embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Chargement base vectorielle
vectordb = Chroma(
    persist_directory="vectordb",
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(
    search_kwargs={"k": 3}
)

def retrieve_context(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return context, docs