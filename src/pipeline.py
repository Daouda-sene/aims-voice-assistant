from src.rag_engine import retrieve_context
from src.llm_engine import generate_answer

def ask_question(question):

    context, docs = retrieve_context(question)

    answer = generate_answer(
        question,
        context
    )

    return answer, docs