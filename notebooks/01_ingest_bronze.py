# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest → Bronze
# MAGIC Downloads the 2023 airline on-time performance dataset from S3 and writes
# MAGIC it as a raw Delta table (bronze layer) in Unity Catalog.

# COMMAND ----------
# MAGIC %pip install boto3 kaggle --quiet

# COMMAND ----------
import boto3
import os
import urllib.request
from pyspark.sql import SparkSession

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET      = "flight-delay-predictor"          # <-- your bucket name
S3_PREFIX      = "raw/flights/"
CATALOG        = "hive_metastore"
BRONZE_DB      = "flight_delay_bronze"
BRONZE_TABLE   = "raw_flights"

# Public mirror of the BTS 2023 On-Time dataset (subset, ~500 k rows)
DATA_URL = (
    "https://raw.githubusercontent.com/jpatokal/openflights/master/"
    "data/routes.dat"
)

# We'll use a pre-cleaned Kaggle-sourced parquet instead (hosted on HuggingFace)
DATASET_URL = (
    "https://huggingface.co/datasets/hepokon365/airline-delay-and-cancellation"
    "/resolve/main/data/2023.csv"
)

LOCAL_PATH = "/tmp/flights_2023.csv"

# COMMAND ----------
# MAGIC %md ## 1 · Download raw data

# COMMAND ----------
print("Downloading flight delay dataset …")
urllib.request.urlretrieve(DATASET_URL, LOCAL_PATH)
print(f"Saved to {LOCAL_PATH}")

# COMMAND ----------
# MAGIC %md ## 2 · Push to S3 (landing zone)

# COMMAND ----------
s3 = boto3.client("s3")
s3_key = f"{S3_PREFIX}flights_2023.csv"

print(f"Uploading to s3://{S3_BUCKET}/{s3_key} …")
s3.upload_file(LOCAL_PATH, S3_BUCKET, s3_key)
print("Upload complete.")

# COMMAND ----------
# MAGIC %md ## 3 · Read from S3 → Bronze Delta table

# COMMAND ----------
spark.sql(f"CREATE DATABASE IF NOT EXISTS {BRONZE_DB}")

df_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"s3://{S3_BUCKET}/{s3_key}")
)

print(f"Row count: {df_raw.count():,}")
df_raw.printSchema()

# COMMAND ----------
(
    df_raw.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{BRONZE_DB}.{BRONZE_TABLE}")
)

print(f"✅  Bronze table written: {BRONZE_DB}.{BRONZE_TABLE}")
display(spark.table(f"{BRONZE_DB}.{BRONZE_TABLE}").limit(5))
