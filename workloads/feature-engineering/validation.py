#!/usr/bin/env python3
"""
Data Validation Script for Mobile Sales Dataset

This script reads raw CSV data from MinIO, performs comprehensive data quality checks,
and fails the pipeline if the data does not meet the defined quality thresholds.

Checks performed:
1. Required column presence (schema validation)
2. Date format validation (must be YYYY-MM-DD)
3. Null values in critical columns
4. Negative values in Price and Quantity Sold (threshold: 1%)
5. Future dates in Dispatch Date (threshold: 0%)
6. Duplicate records based on key columns
"""

import os
import sys
import json
import boto3
import polars as pl
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables 
load_dotenv()

# MinIO configuration (from environment variables)
ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET = os.getenv("BUCKET_NAME", "raw-data")
KEY = os.getenv("BUCKET_KEY", "mobile_sales_data.csv")

# Quality thresholds
MAX_NEGATIVE_PRICE_PCT = 1.0      # If >1% of rows have negative Price -> FAIL
MAX_NEGATIVE_QTY_PCT = 1.0        # If >1% of rows have negative Quantity Sold -> FAIL
MAX_FUTURE_DATES = 0              # If ANY future Dispatch Date -> FAIL
MAX_ALLOWED_NULLS_CRITICAL = 0    # Critical columns (Brand, Quantity Sold) must have 0 nulls
MAX_ALLOWED_NULLS_NON_CRITICAL = 0.1  # Non-critical can have up to 10% nulls

# Expected date format for Dispatch Date (contract)
EXPECTED_DATE_FORMAT = "%Y-%m-%d"

def validate_data() -> dict:
    """
    Validate the raw data from MinIO.
    Returns a dictionary with status, errors, warnings, and stats.
    """
    report = {
        "status": "PASSED",
        "errors": [],
        "warnings": [],
        "stats": {},
        "timestamp": datetime.now().isoformat(),
        "source": f"s3://{BUCKET}/{KEY}"
    }

    print(f"Starting Data Validation for s3://{BUCKET}/{KEY}")

    # --- 1. Load Data from MinIO ---
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=ENDPOINT,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY
        )
        obj = s3.get_object(Bucket=BUCKET, Key=KEY)
        df = pl.read_csv(obj['Body'])
        report["stats"]["total_rows"] = len(df)
        report["stats"]["total_columns"] = len(df.columns)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns from MinIO.")
    except Exception as e:
        report["status"] = "FAILED"
        report["errors"].append(f"Failed to load data from MinIO: {e}")
        return report

    # --- 2. Required Column Presence ---
    required_columns = [
        "Brand", "Quantity Sold", "Dispatch Date", "Region",
        "Price", "Inward Date", "Product Code", "RAM", "ROM"
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        report["status"] = "FAILED"
        report["errors"].append(f"Missing required columns: {missing_columns}")
        return report  # Stop early, schema is broken
    print("All required columns present.")

    # --- 3. Date Format Validation ---
    # Force Dispatch Date to string, then try to parse with expected format
    # If parsing fails, we count how many rows have invalid formats.
    try:
        # Keep a copy of the original Dispatch Date as string
        df = df.with_columns(
            pl.col("Dispatch Date").cast(pl.String).alias("Dispatch_Date_String")
        )
        # Parse the string column into a date
        df = df.with_columns(
            pl.col("Dispatch_Date_String").str.strptime(
                pl.Date, format=EXPECTED_DATE_FORMAT, strict=False
            ).alias("Dispatch Date")
        )
        # Count failed parses
        failed_parse = df.filter(pl.col("Dispatch Date").is_null())
        failed_count = len(failed_parse)
        if failed_count > 0:
            sample_bad = failed_parse.select("Dispatch_Date_String").head(5).to_series().to_list()
            report["status"] = "FAILED"
            report["errors"].append(
                f"Dispatch Date format mismatch: {failed_count} rows failed to parse "
                f"with format '{EXPECTED_DATE_FORMAT}'. Sample bad values: {sample_bad}"
            )
        else:
            print(f"All {len(df)} Dispatch Dates successfully parsed as {EXPECTED_DATE_FORMAT}.")

    except Exception as e:
        report["status"] = "FAILED"
        report["errors"].append(f"Date parsing exception: {e}")
        return report

    # Drop the temporary string column
    df = df.drop("Dispatch_Date_String")    

    # --- 4. Numeric Type Checks & Negative Values ---
    # Quantity Sold
    if df["Quantity Sold"].dtype not in [pl.Int64, pl.Float64]:
        try:
            df = df.with_columns(pl.col("Quantity Sold").cast(pl.Float64))
        except Exception as e:
            report["status"] = "FAILED"
            report["errors"].append(f"Quantity Sold cannot be cast to numeric: {e}")
            return report

    # Price
    if df["Price"].dtype not in [pl.Int64, pl.Float64]:
        try:
            df = df.with_columns(pl.col("Price").cast(pl.Float64))
        except Exception as e:
            report["status"] = "FAILED"
            report["errors"].append(f"Price cannot be cast to numeric: {e}")
            return report

    # Check negative Quantity Sold
    neg_qty_count = df.filter(pl.col("Quantity Sold") < 0).height
    neg_qty_pct = (neg_qty_count / len(df)) * 100
    report["stats"]["negative_qty_count"] = neg_qty_count
    report["stats"]["negative_qty_pct"] = round(neg_qty_pct, 2)

    if neg_qty_count > 0:
        report["warnings"].append(f"Found {neg_qty_count} rows with negative Quantity Sold ({neg_qty_pct:.2f}%)")
        if neg_qty_pct > MAX_NEGATIVE_QTY_PCT:
            report["status"] = "FAILED"
            report["errors"].append(
                f"Negative Quantity Sold exceeds threshold: {neg_qty_pct:.2f}% > {MAX_NEGATIVE_QTY_PCT}%"
            )

    # Check negative Price
    neg_price_count = df.filter(pl.col("Price") < 0).height
    neg_price_pct = (neg_price_count / len(df)) * 100
    report["stats"]["negative_price_count"] = neg_price_count
    report["stats"]["negative_price_pct"] = round(neg_price_pct, 2)

    if neg_price_count > 0:
        report["warnings"].append(f"Found {neg_price_count} rows with negative Price ({neg_price_pct:.2f}%)")
        if neg_price_pct > MAX_NEGATIVE_PRICE_PCT:
            report["status"] = "FAILED"
            report["errors"].append(
                f"Negative Price exceeds threshold: {neg_price_pct:.2f}% > {MAX_NEGATIVE_PRICE_PCT}%"
            )

    # --- 5. Null Value Checks ---
    null_counts = df.null_count()
    critical_cols = ["Brand", "Quantity Sold", "Dispatch Date", "Region"]
    non_critical_cols = ["Customer Name", "Customer Location"]

    for col in critical_cols:
        null_count = null_counts[col][0]
        if null_count > 0:
            report["status"] = "FAILED"
            report["errors"].append(f"Null values found in critical column '{col}': {null_count} rows")

    for col in non_critical_cols:
        null_count = null_counts[col][0]
        if null_count > 0:
            null_pct = (null_count / len(df)) * 100
            if null_pct > MAX_ALLOWED_NULLS_NON_CRITICAL * 100:
                report["warnings"].append(
                    f"High null percentage in '{col}': {null_pct:.2f}% (threshold: {MAX_ALLOWED_NULLS_NON_CRITICAL*100}%)"
                )
            else:
                report["warnings"].append(f"Nulls found in '{col}': {null_count} rows (within acceptable limit)")

    # Store null stats in report
    report["stats"]["null_counts"] = {col: null_counts[col][0] for col in df.columns if null_counts[col][0] > 0}

    # --- 6. Future Date Check ---
    if "Dispatch Date" in df.columns:
        today = datetime.now().date()
        future_rows = df.filter(pl.col("Dispatch Date") > today)
        future_count = len(future_rows)
        report["stats"]["future_dates_count"] = future_count
        if future_count > MAX_FUTURE_DATES:
            report["status"] = "FAILED"
            report["errors"].append(f"Found {future_count} rows with future Dispatch Dates (max allowed: {MAX_FUTURE_DATES})")
        else:
            print("No future Dispatch Dates found.")

    # --- 7. Duplicate Check on Key Columns ---

    key_cols = ["Product Code", "Brand", "Inward Date"]
    dup_groups = df.group_by(key_cols).agg(pl.count()).filter(pl.col("count") > 1)
    if len(dup_groups) > 0:
        total_dup_rows = dup_groups["count"].sum() - len(dup_groups)
        report["warnings"].append(
            f"Found {len(dup_groups)} duplicate groups based on {key_cols}, affecting {total_dup_rows} extra rows."
        )
        report["stats"]["duplicate_groups"] = len(dup_groups)
        report["stats"]["total_duplicate_rows"] = total_dup_rows

    # --- 8. Final Summary Stats ---
    report["stats"]["unique_brands"] = df["Brand"].n_unique()
    report["stats"]["unique_regions"] = df["Region"].n_unique()
    report["stats"]["min_price"] = df["Price"].min()
    report["stats"]["max_price"] = df["Price"].max()
    report["stats"]["min_qty"] = df["Quantity Sold"].min()
    report["stats"]["max_qty"] = df["Quantity Sold"].max()

    # --- 9. Print Final Report ---
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print(f"Status: {report['status']}")
    print(f"Total rows: {report['stats']['total_rows']}")
    print(f"Unique Brands: {report['stats']['unique_brands']}")
    print(f"Unique Regions: {report['stats']['unique_regions']}")

    if report["errors"]:
        print("\nERRORS:")
        for e in report["errors"]:
            print(f"  - {e}")

    if report["warnings"]:
        print("\nWARNINGS:")
        for w in report["warnings"]:
            print(f"  - {w}")

    print("="*60)

    return report


# --- Main Entry Point ---
if __name__ == "__main__":
    report = validate_data()

    # Save report to file 
    with open("/tmp/validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nValidation report saved to /tmp/validation_report.json")

    # Exit with error code if validation failed
    if report["status"] == "FAILED":
        print("Data Validation FAILED. Halting pipeline.")
        sys.exit(1)
    else:
        print("Data Validation PASSED. Proceeding to Feature Engineering.")
        sys.exit(0)