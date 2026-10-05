import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ============================================================
# Конфиг
# ============================================================
CSV_PATH = "2021_german_doctor_reviews.csv"
TEXT_COL = "comment"
LABEL_COL = "rating"
MODEL_NAME = "distilbert-base-german-cased"
SUBSET_SIZE = 80000
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 2
SEED = 42

# ============================================================
# Устройство (CUDA для GTX 1650)
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 1. Загрузка данных
# ============================================================
print("\nЗагрузка данных...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[LABEL_COL])
df = df.sample(n=min(SUBSET_SIZE, len(df)), random_state=SEED).reset_index(drop=True)

# Метки 1-6 -> 0-5
df["label"] = df[LABEL_COL].astype(int) - 1

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED,
    stratify=df["label"]
)

print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print(f"Распределение классов (train):")
print(train_df["label"].value_counts(normalize=True).sort_index())

# ============================================================
# 2. Датасет
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
            str(self.texts[idx]),
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
# 3. Модель и токенизатор
# ============================================================
print(f"\nЗагрузка {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=6
)

# ============================================================
# 4. Датасеты
# ============================================================
train_dataset = ReviewDataset(
    train_df[TEXT_COL].values, train_df["label"].values, tokenizer, MAX_LENGTH
)
test_dataset = ReviewDataset(
    test_df[TEXT_COL].values, test_df["label"].values, tokenizer, MAX_LENGTH
)

# ============================================================
# 5. Метрики
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# ============================================================
# 6. Обучение
# ============================================================
training_args = TrainingArguments(
    output_dir="./distilbert_german_reviews",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    seed=SEED,
    fp16=True,
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

print("\nСтарт обучения...")
start = time.time()
trainer.train()
train_time = time.time() - start
print(f"\nОбщее время обучения: {train_time:.1f}s")

# ============================================================
# 7. Финальная оценка
# ============================================================
print("\nФинальная оценка...")
start = time.time()
results = trainer.evaluate()
inf_time = time.time() - start

print(f"Время инференса: {inf_time:.2f}s")
print(f"Accuracy: {results['eval_accuracy']:.4f}")
print(f"F1 macro: {results['eval_f1_macro']:.4f}")

# Per-class F1
preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=-1)
y_true = preds.label_ids

print("\nОтчёт по классам:")
print(classification_report(y_true, y_pred, digits=4))