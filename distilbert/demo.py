from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL = "VerMichel/german-reviews-distilbert"


def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits.argmax(dim=-1).item() + 1  # model outputs 0-5, convert to 1-6


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    print("Model: VerMichel/german-reviews-distilbert")
    print("Trained on German doctor reviews. Short or ambiguous texts may be misclassified.")
    print("Type a German review, or 'quit' to exit.\n")

    while True:
        text = input("> ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        rating = predict(text, tokenizer, model)
        print(f"Predicted rating: {rating}\n")


if __name__ == "__main__":
    main()