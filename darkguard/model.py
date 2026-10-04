import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATASET_PATH = Path(__file__).resolve().parent / "dataset.tsv"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, sep="\t")
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    df["text"] = df["text"].fillna("")
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def train_model():
    df = load_dataset()
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "data": df,
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(
            y_test,
            y_pred,
            target_names=["Not Dark Pattern", "Dark Pattern"],
            output_dict=True,
            zero_division=0,
        ),
        "feature_count": len(vectorizer.get_feature_names_out()),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "y_test": y_test,
        "y_pred": y_pred,
    }


def predict_text(text: str, model, vectorizer):
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    prediction = int(model.predict(tfidf)[0])
    probabilities = model.predict_proba(tfidf)[0]
    confidence = float(probabilities[prediction])
    return prediction, confidence, probabilities


def explain_prediction(text: str, model, vectorizer, top_n: int = 6):
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    contributions = tfidf.toarray()[0] * model.coef_[0]

    positive_indices = contributions.argsort()[::-1]
    positive_terms = [
        (feature_names[i], float(contributions[i]))
        for i in positive_indices
        if contributions[i] > 0
    ][:top_n]

    negative_indices = contributions.argsort()
    negative_terms = [
        (feature_names[i], float(contributions[i]))
        for i in negative_indices
        if contributions[i] < 0
    ][:top_n]

    return positive_terms, negative_terms
