import pandas as pd
from sklearn.model_selection import train_test_split
from config import CURRENT
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data():
    if CURRENT["source"] == "hf":
        from datasets import load_dataset
        dataset = load_dataset(CURRENT["name"], CURRENT["lang"])
        train = pd.DataFrame(dataset["train"])
        test = pd.DataFrame(dataset["test"])
    elif CURRENT["source"] == "local":
        if CURRENT["test_path"] is None:
            df = pd.read_csv(CURRENT["path"])
            train, test = train_test_split(df, test_size=0.2, random_state=42)
        else:
            train = pd.read_csv(CURRENT["path"])
            test = pd.read_csv(CURRENT["test_path"])
    return train, test

def vectorize(train, test):
    vect = TfidfVectorizer(
        stop_words=stopwords.words('german'),
        max_features=CURRENT["max_features"]
    )
    X_train = vect.fit_transform(train["Text"])
    X_test = vect.transform(test["Text"])
    return X_train, X_test, vect