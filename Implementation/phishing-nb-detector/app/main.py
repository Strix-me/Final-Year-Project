from pathlib import Path
import json
import joblib
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "phishing_nb.joblib"
REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"

STUDENT_NAME = "Abdulrazaq Femi-Sunmonu"
MATRIC_NUMBER = "22120612985"
GRAD_YEAR = "2026"
PROJECT_TITLE = "Naïve Bayes-Based Phishing Email Detection System"

# Precision-focused threshold
THRESHOLD = 0.85

app = FastAPI(title=PROJECT_TITLE)
model = None

app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@app.on_event("startup")
def startup():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Train first: python src/train.py")
    model = joblib.load(MODEL_PATH)


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def interpret(prob: float) -> str:
    if prob >= 0.90:
        return "Very high likelihood of phishing. Avoid clicking links and verify the sender."
    if prob >= 0.70:
        return "High likelihood of phishing. Treat as suspicious and verify via trusted channel."
    if prob >= 0.50:
        return "Moderate likelihood of phishing. Check links and message urgency."
    if prob >= 0.30:
        return "Low likelihood of phishing, but stay cautious."
    return "Very low likelihood of phishing. Email appears legitimate."


@app.get("/", response_class=HTMLResponse)
def home():
    metrics = load_metrics()

    if metrics:
        metrics_html = f"""
        <div class="grid">

          <div class="card">
            <div class="kpi-title">Accuracy</div>
            <div class="kpi">{fmt_pct(metrics.get("accuracy", 0.0))}</div>
            <div class="muted">Overall correctness of predictions.</div>
          </div>

          <div class="card">
            <div class="kpi-title">Precision</div>
            <div class="kpi">{fmt_pct(metrics.get("precision", 0.0))}</div>
            <div class="muted">When flagged as phishing, how often it is correct.</div>
          </div>

          <div class="card">
            <div class="kpi-title">Recall</div>
            <div class="kpi">{fmt_pct(metrics.get("recall", 0.0))}</div>
            <div class="muted">Ability to detect actual phishing emails.</div>
          </div>

          <div class="card">
            <div class="kpi-title">F1 Score</div>
            <div class="kpi">{fmt_pct(metrics.get("f1", 0.0))}</div>
            <div class="muted">Balance between precision and recall.</div>
          </div>

          <div class="card">
            <div class="kpi-title">ROC-AUC</div>
            <div class="kpi">{metrics.get("roc_auc", 0.0):.4f}</div>
            <div class="muted">Model’s ability to distinguish phishing vs legitimate.</div>
          </div>

        <!--
        <div class="card">
          <div class="kpi-title">Threshold</div>
          <div class="kpi" style="color:#6ee7ff;">{metrics.get("threshold", THRESHOLD):.2f}</div>
          <div class="muted">Decision threshold used for classification.</div>
        </div>
        -->

        </div>
        """
    else:
        metrics_html = """
        <div class="card">
          <div class="kpi-title">Metrics not found</div>
          <div class="muted">Train the model first so reports/metrics.json can be generated.</div>
        </div>
        """

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{PROJECT_TITLE}</title>
<style>
:root {{
  --bg: #0b1220;
  --card: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.12);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.65);
}}
body {{
  margin: 0;
  font-family: Arial, sans-serif;
  background: radial-gradient(1200px 800px at 20% 10%, rgba(110,231,255,0.16), transparent 60%),
              radial-gradient(1000px 800px at 90% 30%, rgba(99,102,241,0.14), transparent 60%),
              var(--bg);
  color: var(--text);
}}
.wrap {{
  max-width: 1100px;
  margin: auto;
  padding: 30px 20px 50px;
}}
.header {{
  text-align: center;
  margin-bottom: 28px;
}}
.sub {{
  color: var(--muted);
  font-size: 14px;
}}
.card {{
  border: 1px solid var(--border);
  background: var(--card);
  border-radius: 16px;
  padding: 18px;
  margin-top: 16px;
}}
.grid {{
  display: grid;
  gap: 15px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}}
.chart-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 20px;
  margin-top: 15px;
}}
.kpi-title {{
  font-size: 13px;
  color: var(--muted);
}}
.kpi {{
  font-size: 24px;
  font-weight: bold;
  margin-top: 6px;
}}
.muted {{
  color: var(--muted);
  font-size: 13px;
  margin-top: 8px;
  line-height: 1.35;
}}
textarea {{
  width: 100%;
  height: 190px;
  padding: 12px;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: rgba(0,0,0,0.28);
  color: var(--text);
  resize: none;
  outline: none;
  box-sizing: border-box;
}}
.btn-row {{
  display: flex;
  justify-content: center;
  margin-top: 15px;
}}
button {{
  padding: 11px 28px;
  border-radius: 10px;
  border: 1px solid #6ee7ff;
  background: rgba(110,231,255,0.18);
  color: white;
  font-weight: bold;
  cursor: pointer;
}}
button:hover {{
  background: rgba(110,231,255,0.32);
}}
img {{
  width: 100%;
  border-radius: 10px;
  margin-top: 10px;
  border: 1px solid var(--border);
}}
.section-title {{
  margin-top: 30px;
  margin-bottom: 8px;
}}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>{PROJECT_TITLE}</h1>
    <div class="sub"><b>{STUDENT_NAME}</b> | Matric: {MATRIC_NUMBER} | Class of {GRAD_YEAR}</div>
    <div class="sub">Model: TF-IDF + Multinomial Naïve Bayes + SMOTE + 5-Fold Stratified CV</div>
    <div class="sub">Precision-focused threshold: {THRESHOLD:.2f}</div>
  </div>

  <div class="card">
    <h3>Predict Phishing Email</h3>
    <div class="muted">Paste an email or suspicious message below to classify it as phishing or legitimate.</div>
    <form action="/predict" method="post">
      <textarea name="text" placeholder="Paste email text here..." required></textarea>
      <div class="btn-row">
        <button type="submit">Predict</button>
      </div>
    </form>
  </div>

  <h3 class="section-title">Model Performance</h3>
  {metrics_html}

  <h3 class="section-title">Evaluation Visualizations</h3>
  <div class="chart-grid">
    <div class="card">
      <h4>Confusion Matrix</h4>
      <div class="muted">Shows correct vs incorrect classifications.</div>
      <img src="/reports/confusion_matrix.png" alt="Confusion Matrix">
    </div>

    <div class="card">
      <h4>ROC Curve</h4>
      <div class="muted">Shows the model’s discrimination ability.</div>
      <img src="/reports/roc_curve.png" alt="ROC Curve">
    </div>
  </div>
</div>
</body>
</html>
"""


@app.post("/predict", response_class=HTMLResponse)
def predict(text: str = Form(...)):
    text = text.strip()

    proba = float(model.predict_proba([text])[0][1])
    pred = 1 if proba >= THRESHOLD else 0

    label = "PHISHING" if pred == 1 else "LEGITIMATE"
    explanation = interpret(proba)

    badge_color = "#ef4444" if pred == 1 else "#22c55e"
    bar_width = max(2, min(100, proba * 100))

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Prediction Result</title>
<style>
body {{
  background: #0b1220;
  color: white;
  font-family: Arial, sans-serif;
  margin: 0;
}}
.wrap {{
  max-width: 900px;
  margin: auto;
  padding: 40px 20px;
}}
.card {{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  padding: 20px;
  border-radius: 14px;
  margin-top: 16px;
}}
.badge {{
  display: inline-block;
  padding: 8px 14px;
  border-radius: 999px;
  background: {badge_color};
  color: white;
  font-weight: bold;
}}
.muted {{
  color: rgba(255,255,255,0.70);
  font-size: 14px;
}}
.bar-wrap {{
  margin-top: 14px;
  width: 100%;
  height: 14px;
  background: rgba(255,255,255,0.10);
  border-radius: 999px;
  overflow: hidden;
}}
.bar {{
  height: 100%;
  width: {bar_width:.2f}%;
  background: {badge_color};
}}
pre {{
  white-space: pre-wrap;
  background: rgba(0,0,0,0.25);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 10px;
  padding: 12px;
}}
a {{
  color: #6ee7ff;
  text-decoration: none;
}}
</style>
</head>
<body>
<div class="wrap">
  <h2>Prediction Result</h2>

  <div class="card">
    <p><span class="badge">{label}</span></p>
    <p><b>Phishing Probability:</b> {proba:.4f} ({fmt_pct(proba)})</p>
    <div class="bar-wrap"><div class="bar"></div></div>
    <p class="muted"><b>Interpretation:</b> {explanation}</p>
    <p class="muted"><b>Decision Threshold:</b> {THRESHOLD:.2f}</p>
  </div>

  <div class="card">
    <h3>Input Text</h3>
    <pre>{text}</pre>
  </div>

  <p><a href="/">← Back to Home</a></p>
</div>
</body>
</html>
"""