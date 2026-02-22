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

# ====== EDIT THESE (YOUR DETAILS) ======
STUDENT_NAME = "YOUR FULL NAME"
MATRIC_NUMBER = "YOUR MATRIC NUMBER"
GRAD_YEAR = "2026"  # change to your graduation year
PROJECT_TITLE = "Naïve Bayes-Based Phishing Email Detection System"
# ======================================

app = FastAPI(title=PROJECT_TITLE)
model = None

# Serve reports folder so images can display on the webpage
# This will make /reports/confusion_matrix.png and /reports/roc_curve.png accessible
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
    return f"{x*100:.2f}%"


def interpret(prob: float) -> str:
    # Simple, user-friendly interpretation
    if prob >= 0.90:
        return "Very high likelihood of phishing. Avoid clicking links and verify the sender."
    if prob >= 0.70:
        return "High likelihood of phishing. Treat as suspicious and verify via trusted channel."
    if prob >= 0.50:
        return "Moderate likelihood of phishing. Check links, sender address, and message urgency."
    if prob >= 0.30:
        return "Low likelihood of phishing, but stay cautious with links and attachments."
    return "Very low likelihood of phishing. Email appears legitimate, but always remain cautious."


@app.get("/", response_class=HTMLResponse)
def home():
    metrics = load_metrics()

    metrics_html = ""
    if metrics:
        metrics_html = f"""
        <div class="grid">
          <div class="card">
            <div class="kpi-title">Accuracy</div>
            <div class="kpi">{fmt_pct(metrics.get("accuracy", 0.0))}</div>
            <div class="muted">Overall correctness (both classes).</div>
          </div>
          <div class="card">
            <div class="kpi-title">Precision</div>
            <div class="kpi">{fmt_pct(metrics.get("precision", 0.0))}</div>
            <div class="muted">Of emails flagged as phishing, how many were truly phishing.</div>
          </div>
          <div class="card">
            <div class="kpi-title">Recall</div>
            <div class="kpi">{fmt_pct(metrics.get("recall", 0.0))}</div>
            <div class="muted">Of actual phishing emails, how many were detected.</div>
          </div>
          <div class="card">
            <div class="kpi-title">F1-score</div>
            <div class="kpi">{fmt_pct(metrics.get("f1", 0.0))}</div>
            <div class="muted">Balance between precision and recall.</div>
          </div>
          <div class="card">
            <div class="kpi-title">ROC-AUC</div>
            <div class="kpi">{metrics.get("roc_auc", 0.0):.4f}</div>
            <div class="muted">How well the model separates phishing vs legitimate across thresholds.</div>
          </div>
        </div>
        """
    else:
        metrics_html = """
        <div class="card">
          <b>Metrics not found.</b> Train first to generate <code>reports/metrics.json</code>.
        </div>
        """

    charts_html = f"""
    <div class="grid2">
      <div class="card">
        <h3>Confusion Matrix</h3>
        <div class="muted">Shows correct vs incorrect classifications (Legit vs Phish).</div>
        <img class="img" src="/reports/confusion_matrix.png" alt="Confusion Matrix" />
      </div>
      <div class="card">
        <h3>ROC Curve</h3>
        <div class="muted">Trade-off between True Positive Rate and False Positive Rate.</div>
        <img class="img" src="/reports/roc_curve.png" alt="ROC Curve" />
      </div>
    </div>
    """

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>{PROJECT_TITLE}</title>
  <style>
    :root {{
      --bg: #0b1220;
      --card: rgba(255,255,255,0.06);
      --border: rgba(255,255,255,0.12);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.65);
      --accent: #6ee7ff;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      background: radial-gradient(1200px 800px at 20% 10%, rgba(110,231,255,0.25), transparent 60%),
                  radial-gradient(1000px 800px at 90% 30%, rgba(99,102,241,0.25), transparent 60%),
                  var(--bg);
      color: var(--text);
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 28px 18px 60px; }}
    .header {{
      display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-end;
      justify-content: space-between; margin-bottom: 18px;
    }}
    .title h1 {{ margin: 0; font-size: 26px; letter-spacing: 0.2px; }}
    .title .meta {{ margin-top: 6px; color: var(--muted); }}
    .pill {{
      display:inline-flex; gap:8px; align-items:center;
      padding: 10px 12px; border: 1px solid var(--border); border-radius: 999px;
      background: rgba(255,255,255,0.04);
      color: var(--muted);
      font-size: 13px;
    }}
    .card {{
      border: 1px solid var(--border);
      background: var(--card);
      border-radius: 16px;
      padding: 14px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }}
    .grid {{
      display: grid; gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      margin-top: 14px;
    }}
    .grid2 {{
      display: grid; gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      margin-top: 14px;
    }}
    .kpi-title {{ color: var(--muted); font-size: 13px; }}
    .kpi {{ font-size: 26px; margin-top: 8px; font-weight: 700; }}
    .muted {{ color: var(--muted); font-size: 13px; margin-top: 6px; line-height: 1.35; }}
    textarea {{
      width: 100%; height: 190px; padding: 12px;
      border-radius: 14px; border: 1px solid var(--border);
      background: rgba(0,0,0,0.25); color: var(--text);
      font-size: 14px; outline: none;
    }}
    textarea:focus {{ border-color: rgba(110,231,255,0.5); box-shadow: 0 0 0 3px rgba(110,231,255,0.12); }}
    button {{
      margin-top: 10px;
      border: 1px solid rgba(110,231,255,0.35);
      background: rgba(110,231,255,0.12);
      color: var(--text);
      padding: 10px 16px;
      border-radius: 12px;
      cursor: pointer;
      font-weight: 600;
    }}
    button:hover {{ background: rgba(110,231,255,0.18); }}
    .img {{ width: 100%; border-radius: 12px; margin-top: 10px; border: 1px solid var(--border); background: rgba(255,255,255,0.03); }}
    code {{ color: #d1d5db; }}
    .section-title {{ margin-top: 18px; font-size: 18px; }}
    .footer {{ margin-top: 22px; color: var(--muted); font-size: 12px; }}
  </style>
</head>

<body>
  <div class="wrap">

    <div class="header">
      <div class="title">
        <h1>{PROJECT_TITLE}</h1>
        <div class="meta">
          <b>{STUDENT_NAME}</b> • Matric: <b>{MATRIC_NUMBER}</b> • Graduation Year: <b>{GRAD_YEAR}</b>
        </div>
      </div>
      <div class="pill">Model: <b>TF-IDF + Multinomial Naïve Bayes</b></div>
    </div>

    <div class="card">
      <h3>Predict Phishing Email</h3>
      <div class="muted">
        Paste an email body (or suspicious text). The system returns a classification and a probability score.
      </div>

      <form action="/predict" method="post">
        <textarea name="text" placeholder="Paste email text here..." required></textarea>
        <br/>
        <button type="submit">Predict</button>
      </form>
    </div>

    <div class="section-title">Model Performance Metrics</div>
    {metrics_html}

    <div class="section-title">Evaluation Visualizations</div>
    {charts_html}

    <div class="footer">
      Tip: If images do not show, confirm <code>reports/confusion_matrix.png</code> and <code>reports/roc_curve.png</code> exist.
    </div>

  </div>
</body>
</html>
"""


@app.post("/predict", response_class=HTMLResponse)
def predict(text: str = Form(...)):
    text = text.strip()
    X = [text]

    proba = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])
    label = "PHISHING" if pred == 1 else "LEGITIMATE"
    expl = interpret(proba)

    badge_bg = "rgba(239,68,68,0.18)" if pred == 1 else "rgba(34,197,94,0.18)"
    badge_bd = "rgba(239,68,68,0.35)" if pred == 1 else "rgba(34,197,94,0.35)"

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Prediction Result</title>
  <style>
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      background: #0b1220;
      color: rgba(255,255,255,0.92);
    }}
    .wrap {{ max-width: 900px; margin: 0 auto; padding: 28px 18px 60px; }}
    .card {{
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.06);
      border-radius: 16px;
      padding: 14px;
      margin-top: 12px;
    }}
    .badge {{
      display:inline-block;
      padding: 8px 12px;
      border-radius: 999px;
      background: {badge_bg};
      border: 1px solid {badge_bd};
      font-weight: 700;
    }}
    .muted {{ color: rgba(255,255,255,0.65); font-size: 13px; line-height: 1.35; }}
    pre {{
      white-space: pre-wrap;
      background: rgba(0,0,0,0.25);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      padding: 12px;
      font-size: 13px;
    }}
    a {{ color: #6ee7ff; text-decoration: none; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Prediction Result</h2>

    <div class="card">
      <p><span class="badge">{label}</span></p>
      <p><b>Phishing probability:</b> {proba:.4f} ({fmt_pct(proba)})</p>
      <div class="muted"><b>Interpretation:</b> {expl}</div>
    </div>

    <div class="card">
      <h3>Input</h3>
      <pre>{text}</pre>
    </div>

    <p><a href="/">← Back to Home</a></p>
  </div>
</body>
</html>
"""