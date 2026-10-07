from config import CURRENT
from core.clean import clean

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

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
df = pd.read_csv(CURRENT["csv_path"])
df = df.dropna(subset=[CURRENT["label_col"]])
df = df.sample(n=min(CURRENT["subset_size"], len(df)), random_state=CURRENT["seed"]).reset_index(drop=True)
df["Text"] = df[CURRENT["text_col"]].apply(clean)
df["label"] = df[CURRENT["label_col"]].astype(int) - 1  # 1-6 -> 0-5

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=CURRENT["seed"],
    stratify=df["label"]
)

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ============================================================
# 2. Vocabulary and tokenization (simple, word-based)
# ============================================================
from collections import Counter

def tokenize(text):
    return text.lower().split()

# Train-related glossary
counter = Counter()
for text in train_df["Text"]:
    counter.update(tokenize(text))

# We keep words that appear >=3 times
vocab = {word: idx + 2 for idx, (word, cnt) in enumerate(counter.most_common()) if cnt >= 3}
vocab["<pad>"] = 0
vocab["<unk>"] = 1

print(f"Vocabulary size: {len(vocab)}")

def encode(text, max_length):
    tokens = tokenize(text)[:max_length]
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    # Padding
    if len(ids) < max_length:
        ids += [vocab["<pad>"]] * (max_length - len(ids))
    return ids

# ============================================================
# 3. Dataset
# ============================================================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.max_length)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_dataset = ReviewDataset(train_df["Text"].values, train_df["label"].values, vocab, CURRENT["max_length"])
test_dataset = ReviewDataset(test_df["Text"].values, test_df["label"].values, vocab, CURRENT["max_length"])

train_loader = DataLoader(train_dataset, batch_size=CURRENT["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=CURRENT["batch_size"], shuffle=False)

# ============================================================
# 4. Model
# ============================================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 due to bidirectionality

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        out, (hidden, _) = self.lstm(emb)
        # we take the last hidden state (averaging the two directions)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

model = LSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=CURRENT["embedding_dim"],
    hidden_dim=CURRENT["hidden_dim"],
    num_layers=CURRENT["num_layers"],
    num_classes=6,
    dropout=CURRENT["dropout"],
).to(DEVICE)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 5. Education
# ============================================================
# Class weights (imbalance compensation)
class_counts = train_df["label"].value_counts().sort_index().values
class_weights = len(train_df) / (len(class_counts) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=CURRENT["learning_rate"])

print("\nStart of training...")
start = time.time()

for epoch in range(CURRENT["epochs"]):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{CURRENT['epochs']} — Loss: {total_loss / len(train_loader):.4f}")

train_time = time.time() - start
print(f"\nTraining time: {train_time:.1f}s")

# ============================================================
# 6. Evaluation
# ============================================================
print("\nEvaluation...")
start = time.time()
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        logits = model(input_ids)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

inf_time = time.time() - start

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

labels_sorted = sorted(set(all_labels))
prec, rec, f1, sup = precision_recall_fscore_support(
    all_labels, all_preds, labels=labels_sorted, zero_division=0
)

df_report = pd.DataFrame({
    "Precision": prec,
    "Recall": rec,
    "F1": f1,
    "Support": sup,
}, index=[int(l) + 1 for l in labels_sorted])

print(f"Inference time: {inf_time:.2f}s")
print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print(f"F1 macro: {f1_score(all_labels, all_preds, average='macro'):.4f}")
print()
print(df_report.round(4))