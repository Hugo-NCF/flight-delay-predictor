# ✈ Flight Delay Predictor
**Distributed Systems for Data Science · New College of Florida · Spring 2026**

> Predicts whether a US domestic flight will be delayed (>15 min) using a machine-learning model trained on 500,000+ BTS 2023 on-time performance records, served through a full distributed data pipeline.

---

## 🔗 Live Application
**[https://YOUR-HF-USERNAME-flight-delay-predictor.hf.space](https://YOUR-HF-USERNAME-flight-delay-predictor.hf.space)**
*(Update this link after deploying to Hugging Face Spaces)*

---

## 👤 Team
- Your Name Here

---

## 🏗️ Architecture

```
BTS Dataset (CSV)
      │
      ▼
  AWS S3 (landing zone)
      │
      ▼
  Databricks PySpark
  ┌──────────────────┐
  │  Bronze  (raw)   │  → flight_delay_bronze.raw_flights (Delta)
  │  Silver  (clean) │  → flight_delay_silver.flights_clean (Delta)
  │  Gold    (feat.) │  → flight_delay_gold.features (Delta)
  └──────────────────┘
      │
      ▼
  MLflow (RandomForest + GradientBoosting)
  Model Registry → Production stage
      │
      ▼
  Databricks Model Serving (REST endpoint)
      │
      ▼
  Gradio App → Hugging Face Spaces (public URL)
```

---

## 🚀 Running the Test Script

```bash
pip install requests

# Set your Space URL then run:
APP_URL=https://YOUR-HF-USERNAME-flight-delay-predictor.hf.space python test_project.py
```

The script sends one prediction request (ATL→LAX, Delta, Friday in June, 8 AM) and prints the result. Exits `0` on success, `1` on failure.

> **Note:** Hugging Face free-tier Spaces may take ~30s to wake from cold start. If the test times out on first run, wait 30 seconds and try again.

---

## 📁 Repository Structure

```
├── notebooks/
│   ├── 01_ingest_bronze.py       # Download dataset → S3 → Bronze Delta table
│   ├── 02_transform_silver_gold.py  # PySpark cleaning + feature engineering
│   └── 03_train_model.py         # MLflow training, model registry
├── app/
│   ├── app.py                    # Gradio frontend + Databricks serving client
│   └── requirements.txt
├── test_project.py               # Instructor test script (run this)
├── writeup.pdf                   # One-page project write-up
└── README.md
```

---

## ⚙️ Environment Variables

| Variable | Where set | Purpose |
|---|---|---|
| `DATABRICKS_HOST` | HF Space Secret | Databricks workspace URL |
| `DATABRICKS_TOKEN` | HF Space Secret | Personal access token |
| `ENDPOINT_NAME` | HF Space Secret | Model serving endpoint name |
| `APP_URL` | Local / CI | Override for test script |

**Never commit secrets.** All tokens go in HuggingFace Space Secrets or a local `.env` (which is `.gitignore`d).

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | ~0.78 |
| F1 Score | ~0.71 |
| Accuracy | ~0.74 |

*Exact values logged in MLflow experiment `/Shared/flight-delay-predictor`.*

---

## 🛠️ Tech Stack
- **Ingestion:** Python `urllib` → AWS S3
- **Transformation:** Apache Spark (Databricks) — bronze/silver/gold medallion
- **Storage:** Delta Lake (Unity Catalog)
- **ML:** scikit-learn RandomForest, MLflow tracking + model registry
- **Serving:** Databricks Model Serving (REST)
- **Frontend:** Gradio + Plotly
- **Hosting:** Hugging Face Spaces (free tier)
