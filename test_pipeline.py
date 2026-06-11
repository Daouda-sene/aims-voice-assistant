from src.pipeline import ask_question

answer, docs = ask_question(
    "What is LoRA?"
)

print("\n" + "="*80)
print("ANSWER")
print("="*80)

print(answer)