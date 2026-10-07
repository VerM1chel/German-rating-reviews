from core.config import CURRENT
from core.clean import clean_text

import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    """Load the dataset and split 80/20 for sklearn pipelines (no cleaning, no stratification)."""
    if CURRENT["source"] == "hf":
        from datasets import load_dataset
        dataset = load_dataset(CURRENT["name"], CURRENT["lang"])
        train = pd.DataFrame(dataset["train"])
        test = pd.DataFrame(dataset["test"])
    elif CURRENT["source"] == "local":
        if CURRENT["test_path"] is None:
            df = pd.read_csv(CURRENT["path"])
            train, test = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
            )
        else:
            train = pd.read_csv(CURRENT["path"])
            test = pd.read_csv(CURRENT["test_path"])
    return train, test

def prepare_for_nn():
    """Load, clean, and split data for neural network training (0-indexed labels, stratified split)."""
    df = pd.read_csv(CURRENT["path"])
    df = df.dropna(subset=[CURRENT["label_col"]])
    df = df.sample(
        n=min(CURRENT["subset_size"], len(df)),
        random_state=CURRENT["seed"]
    ).reset_index(drop=True)

    df["Text"] = df[CURRENT["text_col"]].apply(clean_text)
    df["label"] = df[CURRENT["label_col"]].astype(int) - 1  # PyTorch and HuggingFace expect 0-indexed labels

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=CURRENT["seed"],
        stratify=df["label"]
    )
    return train_df, test_df

def vectorize(train, test):
    """TF-IDF vectorization (German stopwords with negations preserved)."""
    german_stops = set(stopwords.words('german'))
    negations = {
        'nicht',
        'kein', 'keine', 'keinen', 'keinem', 'keiner', 'keines',
        'nichts', 'noch', 'ohne',
    }
    german_stops = list(german_stops - negations)
    vect = TfidfVectorizer(
        stop_words=german_stops,
        max_features=CURRENT["max_features"]
    )
    X_train = vect.fit_transform(train["Text"])
    X_test = vect.transform(test["Text"])
    return X_train, X_test, vect