import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Project root
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "emails.csv"
MODEL_PATH = ROOT / "models" / "phishing_nb.joblib"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

# 🔥 IMPORTANT: Your custom threshold
THRESHOLD = 0.85


def main():
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print(df["label"].value_counts())

    X = df["text"].astype(str)
    y = df["label"].astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 1),
            max_features=12000,
            min_df=3,
            max_df=0.85,
            sublinear_tf=True
        )),
        ("smote", SMOTE(k_neighbors=3, random_state=42)),
        ("nb", MultinomialNB(alpha=1.5))
    ])

    # 🔥 Get probabilities (NOT predictions)
    y_proba = cross_val_predict(
        pipeline,
        X,
        y,
        cv=skf,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    # 🔥 Apply threshold manually
    y_pred = (y_proba >= THRESHOLD).astype(int)

    # 🔥 Compute metrics using threshold
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "threshold": THRESHOLD,
        "validation_method": "5-Fold Stratified Cross-Validation",
        "sampling": "SMOTE inside each training fold"
    }

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 📊 Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix (Threshold={THRESHOLD})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Legit", "Phish"])
    plt.yticks([0, 1], ["Legit", "Phish"])

    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # 📈 ROC Curve (unchanged)
    plt.figure()
    RocCurveDisplay.from_predictions(y, y_proba)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=200)
    plt.close()

    # Train final model on full data
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)

    print("\n✅ Threshold-based evaluation completed!")
    print("Threshold:", THRESHOLD)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()