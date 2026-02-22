from __future__ import annotations
import os
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_text(x: str) -> str:
    if x is None:
        return ""
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def load_any_csv(folder: Path, dataset_name: str) -> list[pd.DataFrame]:
    dfs = []
    for p in folder.rglob("*.csv"):
        try:
            df = pd.read_csv(p, encoding_errors="ignore")
            df["__source_file"] = str(p.relative_to(ROOT))
            df["__dataset"] = dataset_name
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
    return dfs

def pick_text_column(df: pd.DataFrame) -> str | None:
    # common column names for email text
    candidates = [
        "text", "body", "email", "content", "message", "Email Text", "EmailText",
        "mail", "mail_text", "Message", "Body", "EmailBody"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: choose the longest average string column
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        return None
    best = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())
    return best

def pick_label_column(df: pd.DataFrame) -> str | None:
    candidates = ["label", "class", "target", "type", "spam", "phishing", "Category", "Label"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def map_labels(series: pd.Series) -> pd.Series:
    # map various label formats to 0/1
    s = series.astype(str).str.lower().str.strip()
    phish_words = {"1", "phish", "phishing", "spam", "malicious", "attack"}
    ham_words = {"0", "ham", "legit", "legitimate", "benign", "normal"}
    out = []
    for v in s:
        if v in phish_words:
            out.append(1)
        elif v in ham_words:
            out.append(0)
        else:
            # try numeric conversion
            try:
                out.append(1 if int(float(v)) == 1 else 0)
            except:
                out.append(None)
    return pd.Series(out)

def build():
    all_rows = []

    # 1) Kaggle
    kaggle_dir = RAW / "kaggle"
    for df in load_any_csv(kaggle_dir, "kaggle"):
        text_col = pick_text_column(df)
        label_col = pick_label_column(df)

        if text_col is None:
            continue

        out = pd.DataFrame()
        out["text"] = df[text_col].map(normalize_text)

        if label_col is not None:
            out["label"] = map_labels(df[label_col])
        else:
            # If your Kaggle file is phishing-only, force label=1
            out["label"] = 1

        out["source"] = df["__dataset"] + ":" + df["__source_file"]
        all_rows.append(out)

    # 2) Enron (often no labels -> force legit)
    enron_dir = RAW / "enron"
    for df in load_any_csv(enron_dir, "enron"):
        text_col = pick_text_column(df)
        if text_col is None:
            continue
        out = pd.DataFrame()
        out["text"] = df[text_col].map(normalize_text)
        out["label"] = 0
        out["source"] = df["__dataset"] + ":" + df["__source_file"]
        all_rows.append(out)

    # 3) USIIL
    usiil_dir = RAW / "usiil"
    for df in load_any_csv(usiil_dir, "usiil"):
        text_col = pick_text_column(df)
        label_col = pick_label_column(df)

        if text_col is None:
            continue

        out = pd.DataFrame()
        out["text"] = df[text_col].map(normalize_text)

        if label_col is not None:
            out["label"] = map_labels(df[label_col])
        else:
            # If you KNOW it’s phishing-only, set 1; if ham-only, set 0.
            # Default: skip if unknown.
            out["label"] = None

        out["source"] = df["__dataset"] + ":" + df["__source_file"]
        all_rows.append(out)

    if not all_rows:
        raise RuntimeError("No CSV files were loaded. Check your data/raw folders.")

    merged = pd.concat(all_rows, ignore_index=True)

    # Drop empty text and unknown labels
    merged = merged[merged["text"].str.len() > 5]
    merged = merged.dropna(subset=["label"])
    merged["label"] = merged["label"].astype(int)

    # Remove duplicates
    merged = merged.drop_duplicates(subset=["text"])

    out_path = OUT_DIR / "emails.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Saved merged dataset: {out_path}")
    print(merged["label"].value_counts())

if __name__ == "__main__":
    build()