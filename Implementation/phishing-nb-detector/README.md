# 🛡️ Naïve Bayes-Based Phishing Email Detection System

## 📖 Overview

This project presents a machine learning–based phishing email detection system developed as a Final Year Project.

The system uses:

- **TF-IDF (Term Frequency–Inverse Document Frequency)** for text feature extraction
- **Multinomial Naïve Bayes** for probabilistic classification
- A **FastAPI web interface** for real-time prediction and visualization

The goal of the system is to automatically classify emails as:

- ✅ Legitimate
- 🚨 Phishing

---

## 👤 Project Information

- **Student Name:** Abdulrazaq Femi-Sunmonu
- **Department Name:** Computer Science
- **Matric Number:** 22120612985
- **Graduation Year:** 2026  
- **Project Type:** Final Year Project  


---

## 🧠 Model Architecture

### 1️⃣ Feature Extraction
- Text preprocessing (lowercasing, stop-word removal)
- N-gram extraction (unigrams and bigrams)
- TF-IDF vectorization

### 2️⃣ Classification Model
- Multinomial Naïve Bayes
- Alpha smoothing applied
- Threshold-based probability decision

---

## 📊 Model Performance

| Metric      | Value |
|-------------|--------|
| Accuracy    | 97.77% |
| Precision   | 78.67% |
| Recall      | 93.61% |
| F1-Score    | 85.49% |
| ROC-AUC     | 0.9954 |

### 📌 Interpretation
- High recall indicates strong phishing detection capability.
- Precision reflects the rate of false positives.
- ROC-AUC near 1.0 demonstrates excellent class separability.

---

## 📈 Evaluation Visualizations

The system generates:

- Confusion Matrix
- ROC Curve
- Metrics JSON report

These are available in:
