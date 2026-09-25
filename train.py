import pandas as pd
from config import CURRENT
from clean import clean
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Очистка
train, test = load_data()
train["Text"] = train[CURRENT["text_col"]].apply(clean)
test["Text"] = test[CURRENT["text_col"]].apply(clean)


vect = TfidfVectorizer(stop_words=CURRENT["stop_words"])
model = LogisticRegression(random_state=42)



# Проверка
print("Train:", train.shape)
print("Test:", test.shape)
print(train["Text"].head(3))
print(train[CURRENT["label_col"]].value_counts())