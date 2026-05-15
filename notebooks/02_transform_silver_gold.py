# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Transform → Silver → Gold
# MAGIC Cleans raw flight data (silver), then engineers features for ML (gold).

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

BRONZE_DB = "flight_delay_bronze"
SILVER_DB = "flight_delay_silver"
GOLD_DB   = "flight_delay_gold"

for db in [SILVER_DB, GOLD_DB]:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")

# COMMAND ----------
# MAGIC %md ## 1 · Silver — clean & type

# COMMAND ----------
df_bronze = spark.table(f"{BRONZE_DB}.raw_flights")

# Standardise column names to snake_case
df_bronze = df_bronze.toDF(*[c.strip().lower().replace(" ", "_")
                              for c in df_bronze.columns])

# Core columns we actually need
KEEP = [
    "month", "day_of_week", "dep_time", "arr_time",
    "crs_dep_time", "crs_arr_time",
    "airline", "origin", "dest",
    "distance",
    "dep_delay", "arr_delay",
    "cancelled", "diverted",
    "carrier_delay", "weather_delay", "nas_delay",
    "security_delay", "late_aircraft_delay",
]

# Keep only columns that exist
existing = [c for c in KEEP if c in df_bronze.columns]
df_silver = df_bronze.select(existing)

# Drop cancelled / diverted flights (no delay concept)
df_silver = df_silver.filter(
    (F.col("cancelled") == 0) & (F.col("diverted") == 0)
)

# Cast numeric columns
num_cols = [
    "month", "day_of_week", "dep_time", "arr_time",
    "crs_dep_time", "crs_arr_time", "distance",
    "dep_delay", "arr_delay",
]
for col in num_cols:
    if col in df_silver.columns:
        df_silver = df_silver.withColumn(col, F.col(col).cast(DoubleType()))

# Drop rows with nulls in key columns
df_silver = df_silver.dropna(subset=["dep_delay", "arr_delay", "distance"])

print(f"Silver row count: {df_silver.count():,}")

(
    df_silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{SILVER_DB}.flights_clean")
)
print(f"✅  Silver table written: {SILVER_DB}.flights_clean")

# COMMAND ----------
# MAGIC %md ## 2 · Gold — feature engineering

# COMMAND ----------
df_gold = spark.table(f"{SILVER_DB}.flights_clean")

# Target: delayed if arrival delay > 15 minutes (FAA definition)
df_gold = df_gold.withColumn(
    "is_delayed", (F.col("arr_delay") > 15).cast(IntegerType())
)

# Time-of-day buckets from scheduled departure
df_gold = df_gold.withColumn(
    "dep_hour", (F.col("crs_dep_time") / 100).cast(IntegerType())
)

df_gold = df_gold.withColumn(
    "time_of_day",
    F.when(F.col("dep_hour") < 6,  "night")
     .when(F.col("dep_hour") < 12, "morning")
     .when(F.col("dep_hour") < 18, "afternoon")
     .otherwise("evening")
)

# Weekend flag
df_gold = df_gold.withColumn(
    "is_weekend",
    F.when(F.col("day_of_week").isin([6, 7]), 1).otherwise(0)
)

# Scheduled flight duration (minutes)
df_gold = df_gold.withColumn(
    "sched_duration",
    ((F.col("crs_arr_time") - F.col("crs_dep_time")) % 2400).cast(IntegerType())
)

# Route-level historical delay rate (aggregated feature)
route_delay = (
    df_gold.groupBy("origin", "dest")
    .agg(F.mean("is_delayed").alias("route_delay_rate"))
)
df_gold = df_gold.join(route_delay, on=["origin", "dest"], how="left")

# Airline-level historical delay rate
airline_delay = (
    df_gold.groupBy("airline")
    .agg(F.mean("is_delayed").alias("airline_delay_rate"))
)
df_gold = df_gold.join(airline_delay, on="airline", how="left")

# Select final feature set
FEATURE_COLS = [
    "month", "day_of_week", "dep_hour", "is_weekend",
    "distance", "sched_duration",
    "route_delay_rate", "airline_delay_rate",
    "is_delayed",          # label
    "airline", "origin", "dest", "time_of_day",  # kept for display
]
existing_features = [c for c in FEATURE_COLS if c in df_gold.columns]
df_gold = df_gold.select(existing_features).dropna()

print(f"Gold row count: {df_gold.count():,}")
delay_rate = df_gold.filter(F.col("is_delayed") == 1).count() / df_gold.count()
print(f"Overall delay rate: {delay_rate:.1%}")

(
    df_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD_DB}.features")
)
print(f"✅  Gold table written: {GOLD_DB}.features")
display(df_gold.limit(5))
