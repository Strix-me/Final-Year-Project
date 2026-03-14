import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
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

    # 5-Fold Stratified Cross-Validation
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Pipeline: TF-IDF -> SMOTE -> Naive Bayes
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
            min_df=2,
            max_df=0.90,
            sublinear_tf=True
        )),
        ("smote", SMOTE(random_state=42)),
        ("nb", MultinomialNB(alpha=0.5))
    ])

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    metrics = {
        "accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "precision_mean": float(np.mean(cv_results["test_precision"])),
        "precision_std": float(np.std(cv_results["test_precision"])),
        "recall_mean": float(np.mean(cv_results["test_recall"])),
        "recall_std": float(np.std(cv_results["test_recall"])),
        "f1_mean": float(np.mean(cv_results["test_f1"])),
        "f1_std": float(np.std(cv_results["test_f1"])),
        "roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
        "roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
        "validation_method": f"{k}-Fold Stratified Cross-Validation",
        "sampling": "SMOTE inside each training fold"
    }

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Cross-validated predictions for plots
    y_pred = cross_val_predict(
        pipeline,
        X,
        y,
        cv=skf,
        method="predict",
        n_jobs=-1
    )

    y_proba = cross_val_predict(
        pipeline,
        X,
        y,
        cv=skf,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Cross-Validated)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Legit", "Phish"])
    plt.yticks([0, 1], ["Legit", "Phish"])

    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # ROC Curve
    plt.figure()
    RocCurveDisplay.from_predictions(y, y_proba)
    plt.title("ROC Curve (Cross-Validated)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=200)
    plt.close()

    # Train final model on full dataset
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"\n✅ {k}-Fold cross-validation completed successfully!")
    print("✅ Saved model:", MODEL_PATH)
    print("✅ Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    main()