# ✈ Flight Delay Predictor
**Distributed Systems for Data Science · New College of Florida · Spring 2026**

> Predicts whether a US domestic flight will be delayed (>15 min) using a machine-learning model trained on 4,078,318 BTS 2022 on-time performance records, served through a full distributed data pipeline.

---

## 🔗 Live Application
**[https://hugoncf-flight-delay-predictor.hf.space](https://hugoncf-flight-delay-predictor.hf.space)**

---

## 👤 Team
- Hugo Cruz

---
## run code
- python test_project.py

## 🏗️ Architecture

```
BTS Combined_Flights_2022.csv (4M rows)
      │
      ▼
  AWS S3 (landing zone)
  s3://flight-delay-predictor/raw/flights/flights_2022.csv
      │
      ▼
  Databricks PySpark (Unity Catalog)
  ┌──────────────────────────────────────────────────────┐
  │  Bronze  (raw)    → workspace.flight_delay_bronze    │
  │  Silver  (clean)  → workspace.flight_delay_silver    │
  │  Gold    (feats)  → workspace.flight_delay_gold      │
  └──────────────────────────────────────────────────────┘
      │
      ▼
  MLflow (RandomForestClassifier)
  Model Registry → workspace.default.flight-delay-clf
      │
      ▼
  Databricks Model Serving (flight-delay-endpoint)
      │
      ▼
  Gradio App → Hugging Face Spaces (public URL)
```

---

## 🚀 Running the Test Script

```bash
pip install gradio_client
python test_project.py
```

The script connects to the live Hugging Face Space, sends one prediction request (ATL→LAX, Delta, Friday in June, 8 AM) and prints the result. Exits `0` on success, `1` on failure.

> **Note:** Hugging Face free-tier Spaces may take ~30s to wake from cold start. If the test times out on first run, wait 30 seconds and try again.

---

## 📁 Repository Structure

```
├── notebooks/
│   ├── 01_ingest_bronze.py          # Download dataset → S3 → Bronze Delta table
│   ├── 02_transform_silver_gold.py  # PySpark cleaning + feature engineering
│   └── 03_train_model.py            # MLflow training, model registry
├── app/
│   ├── app.py                       # Gradio frontend + Databricks serving client
│   └── requirements.txt
├── test_project.py                  # Instructor test script (run this)
├── writeup.pdf                      # One-page project write-up
└── README.md
```

---

## ⚙️ Environment Variables

| Variable | Where set | Purpose |
|---|---|---|
| `DATABRICKS_HOST` | HF Space Secret | Databricks workspace URL |
| `DATABRICKS_TOKEN` | HF Space Secret | Personal access token |
| `ENDPOINT_NAME` | HF Space Secret | Model serving endpoint name |

**Never commit secrets.** All tokens are stored in HuggingFace Space Secrets.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Accuracy | 0.6590 |
| F1 Score | 0.4179 |
| ROC-AUC | 0.6830 |

*Exact values logged in MLflow experiment `/Shared/flight-delay-predictor` in Databricks.*

---

## 🛠️ Tech Stack
- **Ingestion:** Kaggle API → AWS S3
- **Transformation:** Apache Spark (Databricks) — bronze/silver/gold medallion architecture
- **Storage:** Delta Lake (Unity Catalog)
- **ML:** scikit-learn RandomForestClassifier, MLflow tracking + model registry
- **Serving:** Databricks Model Serving (REST endpoint)
- **Frontend:** Gradio + Plotly
- **Hosting:** Hugging Face Spaces (free tier)
