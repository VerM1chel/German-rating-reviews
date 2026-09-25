from config import CURRENT
from data_loader import load_data
from clean import clean

import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


train, test = load_data()
# Проверка
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print(f"Train gaps: {train.isna().sum()}")
print(f"Test gaps: {test.isna().sum()}")
# Очистка
train["Text"] = train[CURRENT["text_col"]].apply(clean)
test["Text"] = test[CURRENT["text_col"]].apply(clean)
train = train.dropna(subset=[CURRENT["label_col"]])
test = test.dropna(subset=[CURRENT["label_col"]])

print(train["Text"].head(3))
print(train[CURRENT["label_col"]].value_counts())


vect = TfidfVectorizer(stop_words=stopwords.words('german'))
X_train = vect.fit_transform(train["Text"])
y_train = train[CURRENT["label_col"]]
X_test = vect.transform(test["Text"])


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_test = model.predict(X_test)
accuracy = accuracy_score(y_test, test[CURRENT["label_col"]])
print(f"Accuracy: {accuracy_score(test[CURRENT["label_col"]], y_test):.4f}")
print(f"F1 macro: {f1_score(test[CURRENT["label_col"]], y_test, average='macro'):.4f}")