from core.data_loader import prepare_for_nn

import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from distilbert.config import CURRENT


# ============================================================
# Device
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 1. Loading data
# ============================================================
print("\nLoading data...")
train_df, test_df = prepare_for_nn()
print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print(f"Class distribution (train):")
print(train_df["label"].value_counts(normalize=True).sort_index())

# ============================================================
# 2. Dataset
# ============================================================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),  # cast to str: original data may contain NaN
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ============================================================
# 3. Model and tokenizer
# ============================================================
print(f"\nLoading {CURRENT['model_name']}...")
tokenizer = AutoTokenizer.from_pretrained(CURRENT["model_name"])
model = AutoModelForSequenceClassification.from_pretrained(
    CURRENT["model_name"], num_labels=6
)

# ============================================================
# 4. Datasets
# ============================================================
train_dataset = ReviewDataset(
    train_df["Text"].values, train_df["label"].values, tokenizer, CURRENT["max_length"]
)
test_dataset = ReviewDataset(
    test_df["Text"].values, test_df["label"].values, tokenizer, CURRENT["max_length"]
)

# ============================================================
# 5. Metrics
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# ============================================================
# 6. Training
# ============================================================
training_args = TrainingArguments(
    output_dir=CURRENT["output_dir"],
    num_train_epochs=CURRENT["epochs"],
    per_device_train_batch_size=CURRENT["batch_size"],
    per_device_eval_batch_size=CURRENT["batch_size"],
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    seed=CURRENT["seed"],
    fp16=True,  # ~2x faster on GTX 1650, no accuracy loss observed
    dataloader_num_workers=0,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("\nStart of training...")
start = time.time()
trainer.train()
train_time = time.time() - start
print(f"\nTraining time: {train_time:.1f}s")

# ============================================================
# 7. Evaluation
# ============================================================
print("\nEvaluation...")
start = time.time()
results = trainer.evaluate()
inf_time = time.time() - start

print(f"Inference time: {inf_time:.2f}s")
print(f"Accuracy: {results['eval_accuracy']:.4f}")
print(f"F1 macro: {results['eval_f1_macro']:.4f}")

# Per-class F1
preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=-1)
y_true = preds.label_ids

labels_sorted = sorted(set(y_true))
prec, rec, f1, sup = precision_recall_fscore_support(
    y_true, y_pred, labels=labels_sorted, zero_division=0
)

df_report = pd.DataFrame({
    "Precision": prec,
    "Recall": rec,
    "F1": f1,
    "Support": sup,
}, index=[int(l) + 1 for l in labels_sorted])

print()
print(df_report.round(4))