#!/usr/bin/env python3
"""
Model Evaluation Script

This script:
1. Loads the FIXED test set from MinIO.
2. Loads the NEW model (from the latest MLflow run).
3. Loads the CHAMPION model (from the 'champion' alias, if exists).
4. Evaluates BOTH models on the same test set.
5. Logs the test metrics back to the NEW model's MLflow run.
6. Compares metrics using business-defined thresholds.
7. Sets the 'champion' alias if the new model is better.
8. Writes a decision.json for Argo to capture.
"""

import os
import sys
import io
import json
import boto3
import mlflow
import polars as pl
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

load_dotenv()

# ==================== CONFIGURATION ====================
ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Input: Test data 
DATA_BUCKET = os.getenv("OUTPUT_BUCKET", "processed-features")
TEST_DATA_KEY = os.getenv("TEST_DATA_KEY", "test_data.parquet")

# MLflow
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-mlflow.mldata.svc.cluster.local:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "mobile_sales_prediction")
MODEL_NAME = os.getenv("MODEL_NAME", "mobile-sales-predictor")
ALIAS_CHAMPION = os.getenv("ALIAS_CHAMPION", "champion")

# Data
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "Quantity Sold")

# Decision Thresholds (Business-defined)
R2_IMPROVEMENT_THRESHOLD = float(os.getenv("R2_THRESHOLD", 0.01))    # Must improve R² by 1%

print("=" * 60)
print("Starting Model Evaluation")
print("=" * 60)
print(f"Test Data: s3://{DATA_BUCKET}/{TEST_DATA_KEY}")
print(f"MLflow:    {MLFLOW_URI}")
print(f"Model:     {MODEL_NAME} (champion alias: {ALIAS_CHAMPION})")
print(f"Thresholds: R² > {R2_IMPROVEMENT_THRESHOLD}")
print("=" * 60)

# ==================== INITIALIZE CLIENTS ====================
s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

mlflow.set_tracking_uri(MLFLOW_URI)
client = mlflow.tracking.MlflowClient()

# ==================== LOAD FIXED TEST DATA ====================
print(f"\nLoading test data from s3://{DATA_BUCKET}/{TEST_DATA_KEY}...")
try:
    obj = s3.get_object(Bucket=DATA_BUCKET, Key=TEST_DATA_KEY)
    data = io.BytesIO(obj['Body'].read())
    test_df = pl.read_parquet(data)
    print(f"Loaded {len(test_df)} test records.")
except Exception as e:
    print(f"Failed to load test data: {e}")
    sys.exit(1)

# Prepare X and y
X_test = test_df.drop(TARGET_COLUMN).to_pandas()
y_test = test_df[TARGET_COLUMN].to_pandas()
print(f"Test set features: {list(X_test.columns)}")

# ==================== EVALUATION HELPER ====================
def evaluate_model(model, X, y):
    """Evaluate a model and return its metrics."""
    y_pred = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": float(mean_squared_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }

# ==================== LOAD THE NEW MODEL ====================
print(f"\nFetching the latest run from experiment '{EXPERIMENT_NAME}'")
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    print(f"Experiment '{EXPERIMENT_NAME}' not found.")
    sys.exit(1)

versions = client.search_model_versions(f"name='{MODEL_NAME}'")

if not versions:
    print(f"No versions found for model '{MODEL_NAME}'.")
    sys.exit(1)

latest_version = max(versions, key=lambda v: int(v.version))
RUN_ID = latest_version.run_id
print(f"Latest run ID: {RUN_ID} (version {latest_version.version})")

# Load the model from the run
try:
    new_model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")
    print("New model loaded.")
except Exception as e:
    print(f"Failed to load new model: {e}")
    sys.exit(1)

new_metrics = evaluate_model(new_model, X_test, y_test)
print(f"\nNew Model Test Metrics:")
print(f"   MAE:  {new_metrics['mae']:.4f}")
print(f"   RMSE: {new_metrics['rmse']:.4f}")
print(f"   R²:   {new_metrics['r2']:.4f}")

# ==================== LOG TEST METRICS BACK TO THE RUN ====================

with mlflow.start_run(run_id=RUN_ID):
    mlflow.log_metric("test_mae", new_metrics["mae"])
    mlflow.log_metric("test_rmse", new_metrics["rmse"])
    mlflow.log_metric("test_r2", new_metrics["r2"])
print(f"Test metrics logged to MLflow run: {RUN_ID}")

# ==================== LOAD THE CHAMPION MODEL ====================
champion_model = None
champion_metrics = None

print(f"\nLooking for existing champion (alias='{ALIAS_CHAMPION}')")
try:
    champion_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_CHAMPION)
    champion_model = mlflow.sklearn.load_model(f"runs:/{champion_version.run_id}/model")
    champion_metrics = evaluate_model(champion_model, X_test, y_test)
    print(f"Champion found: version={champion_version.version}, run={champion_version.run_id}")
    print(f"\nChampion Test Metrics:")
    print(f"MAE:  {champion_metrics['mae']:.4f}")
    print(f"RMSE: {champion_metrics['rmse']:.4f}")
    print(f"R²:   {champion_metrics['r2']:.4f}")
except Exception as e:
    print(f"No champion found (alias '{ALIAS_CHAMPION}' not set). This is the first run.")

# ==================== COMPARE AND DECIDE ====================
PROMOTE = False
DECISION_REASON = ""

if champion_model is None:
    PROMOTE = True
    DECISION_REASON = "No champion exists. Promoting the first model."
else:
    r2_improvement = new_metrics["r2"] - champion_metrics["r2"]

    r2_ok = r2_improvement > R2_IMPROVEMENT_THRESHOLD


    if r2_ok:
        PROMOTE = True
        DECISION_REASON = (
            f"R² improved by {r2_improvement:.4f} (>{R2_IMPROVEMENT_THRESHOLD})"
        )
    else:
        reasons = []
        if not r2_ok:
            reasons.append(f"R² improvement ({r2_improvement:.4f}) below threshold ({R2_IMPROVEMENT_THRESHOLD}).")
        DECISION_REASON = " ".join(reasons)

# ==================== PRINT DECISION ====================
print("\n" + "=" * 60)
print("PROMOTION DECISION")
print("=" * 60)
print(f"New Model R²: {new_metrics['r2']:.4f}")
if champion_model:
    print(f"Champion R²: {champion_metrics['r2']:.4f}")
    print(f"R² Improvement: {new_metrics['r2'] - champion_metrics['r2']:.4f}")
print(f"Decision:{'PROMOTE' if PROMOTE else 'DO NOT PROMOTE'}")
print(f"Reason:{DECISION_REASON}")
print("=" * 60)

# ==================== SET CHAMPION ALIAS IF PROMOTED ====================
if PROMOTE:
    print(f"\nSetting alias '{ALIAS_CHAMPION}' to the new model.")
    # Find the model version that corresponds to RUN_ID
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    target_version = None
    for v in versions:
        if v.run_id == RUN_ID:
            target_version = v.version
            break

    if target_version:
        client.set_registered_model_alias(MODEL_NAME, ALIAS_CHAMPION, target_version)
        print(f"Alias '{ALIAS_CHAMPION}' set to version {target_version}.")
    else:
        print(f"Could not find a registered model version for run {RUN_ID}.")
        sys.exit(1)
else:
    print(f"\n Promotion not approved. Alias '{ALIAS_CHAMPION}' remains on the existing version.")

# ==================== SAVE DECISION FOR ARGO ====================
decision = {
    "timestamp": datetime.now().isoformat(),
    "promote": PROMOTE,
    "reason": DECISION_REASON,
    "new_run_id": RUN_ID,
    "new_metrics": new_metrics,
    "champion_metrics": champion_metrics,
    "model_name": MODEL_NAME,
    "champion_alias": ALIAS_CHAMPION,
}

with open("/tmp/decision.json", "w") as f:
    json.dump(decision, f, indent=2)

print("\n Decision saved to /tmp/decision.json")
print(json.dumps(decision, indent=2))
print("\n" + "=" * 60)
print("Evaluation completed successfully!")
print("=" * 60)
sys.exit(0)