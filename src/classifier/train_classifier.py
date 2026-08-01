"""
Module 1, step 2 — train the notice classifier.

Reads your hand-labeled CSV (data/labels/labeled_notices.csv, produced by
label_notices.py after you fill in the 'category' column), trains a
TF-IDF + Logistic Regression pipeline, evaluates it, and saves the trained
pipeline to disk.

Usage:
    python -m src.classifier.train_classifier
"""
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import LABELED_DATA_PATH, CLASSIFIER_MODEL_PATH, CONFUSION_MATRIX_PATH, MODELS_DIR


def _load_labeled_data() -> pd.DataFrame:
    if not os.path.exists(LABELED_DATA_PATH):
        raise RuntimeError(
            f"No labeled data found at {LABELED_DATA_PATH}. Run "
            "label_notices.py first, fill in the 'category' column, and "
            "save the file at that path."
        )
    df = pd.read_csv(LABELED_DATA_PATH)
    df = df.dropna(subset=["category"])
    df = df[df["category"].str.strip() != ""]
    if len(df) < 10:
        raise RuntimeError(
            f"Only {len(df)} labeled rows found. Label at least ~30-50 "
            "notices for a meaningful train/test split."
        )
    df["text"] = (df["subject_line"].fillna("") + " " + df["main_body_content"].fillna(""))
    return df


def train_and_evaluate() -> Pipeline:
    df = _load_labeled_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred))

    labels = sorted(df["category"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    os.makedirs(MODELS_DIR, exist_ok=True)
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    print(f"\nConfusion matrix saved to {CONFUSION_MATRIX_PATH}")

    joblib.dump(pipeline, CLASSIFIER_MODEL_PATH)
    print(f"Trained model saved to {CLASSIFIER_MODEL_PATH}")

    return pipeline


if __name__ == "__main__":
    train_and_evaluate()
