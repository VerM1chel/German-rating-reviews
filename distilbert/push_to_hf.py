from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LOCAL_PATH = str(Path(__file__).parent / "distilbert_output" / "checkpoint-8000")
REPO_NAME = "VerMichel/german-reviews-distilbert"

print(f"Path: {LOCAL_PATH}")
print(f"Exists: {Path(LOCAL_PATH).exists()}")
print(f"Files: {list(Path(LOCAL_PATH).iterdir())}")

model = AutoModelForSequenceClassification.from_pretrained(LOCAL_PATH)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-german-cased")

model.push_to_hub(REPO_NAME)
tokenizer.push_to_hub(REPO_NAME)

print(f"Готово: https://huggingface.co/{REPO_NAME}")