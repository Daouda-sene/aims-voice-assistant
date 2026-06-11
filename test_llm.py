from src.llm_engine import generate_answer

answer = generate_answer(
    "What is LoRA?",
    "LoRA is a parameter efficient fine tuning method."
)

print(answer)