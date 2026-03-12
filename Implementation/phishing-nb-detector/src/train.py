import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)
from imblearn.over_sampling import SMOTE

# Project root
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "emails.csv"
MODEL_PATH = ROOT / "models" / "phishing_nb.joblib"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)
    print("Original class distribution:")
    print(df["label"].value_counts())

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    X = df["text"]
    y = df["label"]

    # 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    print("\nTraining set size:", X_train.shape[0])
    print("Testing set size:", X_test.shape[0])

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        max_df=0.90,
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("\nBefore SMOTE:")
    print(pd.Series(y_train).value_counts())

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_vec, y_train)

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_smote).value_counts())

    classifier = MultinomialNB(alpha=0.5)
    classifier.fit(X_train_smote, y_train_smote)

    threshold = 0.75
    y_proba = classifier.predict_proba(X_test_vec)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "threshold": threshold,
        "train_test_split": "70/30",
        "sampling": "SMOTE on training set only"
    }

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Legit", "Phish"])
    plt.yticks([0, 1], ["Legit", "Phish"])

    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    plt.figure()
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=200)
    plt.close()

    # Save plain objects, not a custom class
    bundle = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "threshold": threshold
    }
    joblib.dump(bundle, MODEL_PATH)

    print("\n✅ Model trained successfully with SMOTE and 70/30 split!")
    print("✅ Saved model:", MODEL_PATH)
    print("✅ Metrics:", metrics)


if __name__ == "__main__":
    main()