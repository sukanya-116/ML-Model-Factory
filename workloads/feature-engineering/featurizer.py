#!/usr/bin/env python3
"""
Feature Engineering Script for Mobile Sales Dataset

This script reads the raw CSV from MinIO, applies feature transformations,
and writes the processed features as a Parquet file back to MinIO.

Feature Engineering Steps:
1. Filter out negative Price and Quantity Sold (data cleaning)
2. Parse dates (Dispatch Date and Inward Date)
3. Extract date-based features: year, month, day of week
4. Compute days_to_sell (Dispatch Date - Inward Date)
5. Create Revenue = Price * Quantity Sold
6. Add aggregated features: avg_price_per_brand, avg_qty_per_region
7. Categorical encoding: brand_code, region_code, ram_code, rom_code
8. Text feature: product_spec_length
9. Handle nulls (fill with 0)
10. Save to MinIO as Parquet
"""
import os
import sys
import json
import boto3
import polars as pl
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# MinIO configuration
ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
INPUT_BUCKET = os.getenv("INPUT_BUCKET", "raw-data")
INPUT_KEY = os.getenv("INPUT_KEY", "mobile_sales_data.csv")
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET", "processed-features")
RUN_DATE = os.getenv("RUN_DATE", datetime.now().strftime("%Y-%m-%d"))

def feature_engineering():
    """Main feature engineering pipeline."""
    print(f"Starting Feature Engineering for s3://{INPUT_BUCKET}/{INPUT_KEY}")
    
    # --- 1. Initialize S3 client ---
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )

    # --- 2. Load raw data ---
    try:
        obj = s3.get_object(Bucket=INPUT_BUCKET, Key=INPUT_KEY)
        df = pl.read_csv(obj['Body'])
        print(f"Loaded {len(df)} records from MinIO.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    # --- 3. Data Cleaning (filter out invalid rows) ---

    original_count = len(df)
    df = df.filter(
        (pl.col("Price") >= 0) &
        (pl.col("Quantity Sold") >= 0)
    )
    filtered_count = original_count - len(df)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} rows with negative values.")

    # --- 4. Parse Dates ---
    # Dispatch Date is already in YYYY-MM-DD from validation
    df = df.with_columns([
        pl.col("Dispatch Date").cast(pl.String).str.strptime(pl.Date, format="%Y-%m-%d").alias("Dispatch Date"),
        pl.col("Inward Date").cast(pl.String).str.strptime(pl.Date, format="%Y-%m-%d").alias("Inward Date")
    ])

    # --- 5. Feature Engineering Steps ---

    # A) Time-based features (seasonality)
    df = df.with_columns([
        pl.col("Dispatch Date").dt.year().alias("dispatch_year"),
        pl.col("Dispatch Date").dt.month().alias("dispatch_month"),
        pl.col("Dispatch Date").dt.weekday().alias("dispatch_day_of_week"),  # Monday=1, Sunday=7
        pl.col("Dispatch Date").dt.day().alias("dispatch_day"),
    ])

    # B) Inventory turnover (days to sell)
    df = df.with_columns(
        (pl.col("Dispatch Date") - pl.col("Inward Date")).dt.total_days().alias("days_to_sell")
    )

    # C) Revenue (derived metric)
    df = df.with_columns(
        (pl.col("Price") * pl.col("Quantity Sold")).alias("Revenue")
    )

    # D) Aggregated features (added context per group)
    # Average Price per Brand
    brand_avg_price = df.group_by("Brand").agg(
        pl.col("Price").mean().alias("avg_price_per_brand")
    )
    df = df.join(brand_avg_price, on="Brand", how="left")

    # Average Quantity Sold per Region
    region_avg_qty = df.group_by("Region").agg(
        pl.col("Quantity Sold").mean().alias("avg_qty_per_region")
    )
    df = df.join(region_avg_qty, on="Region", how="left")

    # E) Categorical Encoding (convert to numeric codes)
    df = df.with_columns([
        pl.col("Brand").cast(pl.Categorical).to_physical().alias("brand_code"),
        pl.col("Region").cast(pl.Categorical).to_physical().alias("region_code"),
        pl.col("RAM").cast(pl.Categorical).to_physical().alias("ram_code"),
        pl.col("ROM").cast(pl.Categorical).to_physical().alias("rom_code"),
    ])

    # F) Text feature: length of Product Specification
    df = df.with_columns(
        pl.col("Product Specification").str.len_chars().alias("spec_length")
    )

    # --- 6. Select final feature set ---
    final_columns = [
        # Target variable (what we want to predict)
        "Quantity Sold",
        # Numeric/Date features
        "Price",
        "days_to_sell",
        "Revenue",
        "dispatch_year",
        "dispatch_month",
        "dispatch_day_of_week",
        "spec_length",
        # Encoded categoricals
        "brand_code",
        "region_code",
        "ram_code",
        "rom_code",
        # Aggregated features (added context)
        "avg_price_per_brand",
        "avg_qty_per_region",
    ]

    # Ensure all columns exist before selecting
    missing_cols = [c for c in final_columns if c not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        sys.exit(1)

    feature_df = df.select(final_columns)

    # --- 7. Handle Nulls ---
    # Fill any remaining nulls with 0
    feature_df = feature_df.fill_null(0)

    print(f"Feature engineering complete. Shape: {feature_df.shape}")
    print(f"Columns: {feature_df.columns}")

    # --- 8. Save to MinIO as Parquet ---
    output_key = f"features_{RUN_DATE}.parquet"
    local_path = f"/tmp/{output_key}"
    feature_df.write_parquet(local_path)

    # Ensure output bucket exists
    try:
        s3.head_bucket(Bucket=OUTPUT_BUCKET)
    except:
        s3.create_bucket(Bucket=OUTPUT_BUCKET)
        print(f"Created bucket: {OUTPUT_BUCKET}")

    # Upload to MinIO
    try:
        s3.upload_file(
            local_path,
            OUTPUT_BUCKET,
            output_key,
            ExtraArgs={"ContentType": "application/parquet"}
        )
        print(f"Uploaded features to s3://{OUTPUT_BUCKET}/{output_key}")
    except Exception as e:
        print(f"Failed to upload to MinIO: {e}")
        sys.exit(1)

    # --- 9. Write output URI for Argo to capture ---
    uri = f"s3://{OUTPUT_BUCKET}/{output_key}"
    with open("/tmp/output_uri.txt", "w") as f:
        f.write(uri)
    print(f"Output URI saved: {uri}")

    print("Feature Engineering completed successfully!")
    sys.exit(0)

if __name__ == "__main__":
    feature_engineering()