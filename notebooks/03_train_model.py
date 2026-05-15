# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Train Model → MLflow Registry
# MAGIC Trains a RandomForestClassifier on the gold feature table,
# MAGIC logs everything to MLflow, and registers the best model.

# COMMAND ----------
# MAGIC %pip install scikit-learn imbalanced-learn --quiet

# COMMAND ----------
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

GOLD_DB     = "flight_delay_gold"
EXPERIMENT  = "/Shared/flight-delay-predictor"
MODEL_NAME  = "flight-delay-clf"

mlflow.set_experiment(EXPERIMENT)

# COMMAND ----------
# MAGIC %md ## 1 · Load gold features

# COMMAND ----------
df = spark.table(f"{GOLD_DB}.features").toPandas()
print(f"Loaded {len(df):,} rows")
print(df["is_delayed"].value_counts(normalize=True).to_string())

# COMMAND ----------
# MAGIC %md ## 2 · Prepare features

# COMMAND ----------
NUMERIC_FEATURES = [
    "month", "day_of_week", "dep_hour", "is_weekend",
    "distance", "sched_duration",
    "route_delay_rate", "airline_delay_rate",
]

# Encode categorical airline as numeric for RF
le_airline = LabelEncoder()
le_origin  = LabelEncoder()
le_dest    = LabelEncoder()

df["airline_enc"] = le_airline.fit_transform(df["airline"].astype(str))
df["origin_enc"]  = le_origin.fit_transform(df["origin"].astype(str))
df["dest_enc"]    = le_dest.fit_transform(df["dest"].astype(str))

FEATURES = NUMERIC_FEATURES + ["airline_enc", "origin_enc", "dest_enc"]
TARGET   = "is_delayed"

X = df[FEATURES].fillna(0)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

# COMMAND ----------
# MAGIC %md ## 3 · Train & log models

# COMMAND ----------
def log_model(model, name, params):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size",  len(X_test))
        mlflow.log_param("features",   FEATURES)

        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)

        mlflow.log_metrics({
            "accuracy":  acc,
            "f1":        f1,
            "roc_auc":   auc,
            "precision": prec,
            "recall":    rec,
        })

        # Log feature importances if available
        if hasattr(model, "feature_importances_"):
            fi = dict(zip(FEATURES, model.feature_importances_))
            mlflow.log_dict(fi, "feature_importances.json")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_test.iloc[:3],
        )

        run_id = mlflow.active_run().info.run_id
        print(f"{name:40s}  acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")
        return run_id, auc, model

# Random Forest (primary)
rf, rf_params = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    class_weight="balanced", random_state=42, n_jobs=-1
), {
    "model_type": "RandomForest",
    "n_estimators": 200, "max_depth": 10, "class_weight": "balanced"
}

# Gradient Boosting (challenger)
gb, gb_params = GradientBoostingClassifier(
    n_estimators=150, max_depth=5,
    learning_rate=0.1, random_state=42
), {
    "model_type": "GradientBoosting",
    "n_estimators": 150, "max_depth": 5, "learning_rate": 0.1
}

results = []
for model, name, params in [
    (rf, "RandomForest", rf_params),
    (gb, "GradientBoosting", gb_params),
]:
    results.append(log_model(model, name, params))

# COMMAND ----------
# MAGIC %md ## 4 · Promote best model to Production

# COMMAND ----------
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Find the version with highest ROC-AUC just registered
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
best_version = max(versions, key=lambda v: float(
    client.get_run(v.run_id).data.metrics.get("roc_auc", 0)
))

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=best_version.version,
    stage="Production",
    archive_existing_versions=True,
)

print(f"✅  Model '{MODEL_NAME}' v{best_version.version} promoted to Production")
print(f"    ROC-AUC: {client.get_run(best_version.run_id).data.metrics['roc_auc']:.4f}")

# COMMAND ----------
# MAGIC %md ## 5 · Save encoder mappings for serving

# COMMAND ----------
import json

encoder_map = {
    "airlines":  list(le_airline.classes_),
    "origins":   list(le_origin.classes_),
    "dests":     list(le_dest.classes_),
    "features":  FEATURES,
}

with open("/tmp/encoder_map.json", "w") as f:
    json.dump(encoder_map, f)

# Upload to S3 so the serving layer can access it
import boto3
boto3.client("s3").upload_file(
    "/tmp/encoder_map.json",
    "flight-delay-predictor",   # <-- your bucket
    "model/encoder_map.json"
)
print("✅  Encoder map saved to S3")

# Also display classification report
best_model = [r[2] for r in results if r[1] == max(r[1] for r in results)][0]
y_pred_best = best_model.predict(X_test)
print("\nClassification Report (best model):")
print(classification_report(y_test, y_pred_best, target_names=["On-Time", "Delayed"]))
