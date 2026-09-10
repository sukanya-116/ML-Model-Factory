#!/usr/bin/env python3
"""
Model Training Script with Optuna Hyperparameter Optimization

This script:
1. Loads the processed feature Parquet file from MinIO.
2. Splits data into Train (60%), Validation (20%), and Test (20%).
3. Runs an Optuna study to find the best RandomForest hyperparameters.
4. Trains a final model on Train + Validation using the best params.
5. Registers the model in MLflow.
6. Saves the test data and the model URI for downstream stages.
"""

import os
import sys
import io
import json
import boto3
import polars as pl
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
from dotenv import load_dotenv
from datetime import datetime

optuna.logging.set_verbosity(optuna.logging.WARNING)

load_dotenv()

ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET", "processed-features")
FEATURES_KEY = os.getenv("FEATURES_KEY", "features_latest.parquet")
TEST_DATA_KEY = os.getenv("TEST_DATA_KEY", "test_data.parquet")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-mlflow.mldata.svc.cluster.local:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "mobile_sales_prediction")
MODEL_NAME = os.getenv("MODEL_NAME", "mobile-sales-predictor")

TARGET_COLUMN = os.getenv("TARGET_COLUMN", "Quantity Sold")

N_TRIALS = int(os.getenv("OPTUNA_TRIALS", 10))
TIMEOUT = int(os.getenv("OPTUNA_TIMEOUT", 600))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

print("=" * 60)
print("Starting Model Training with Optuna")
print("=" * 60)
print(f"Output: s3://{OUTPUT_BUCKET}/{TEST_DATA_KEY}")
print(f"MLflow: {MLFLOW_URI}")
print(f"Target: {TARGET_COLUMN}")
print(f"Optuna: {N_TRIALS} trials, timeout={TIMEOUT}s")
print("=" * 60)

# ==================== INITIALIZE S3 CLIENT ====================
s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

# ==================== LOAD PROCESSED FEATURES ====================
print(f"\nLoading features from s3://{OUTPUT_BUCKET}/{FEATURES_KEY}")
try:
    obj = s3.get_object(Bucket=OUTPUT_BUCKET, Key=FEATURES_KEY)
    data = io.BytesIO(obj['Body'].read())
    df = pl.read_parquet(data)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns.")
    print(f"Columns: {df.columns}")
except Exception as e:
    print(f"Failed to load features: {e}")
    sys.exit(1)

# ==================== SPLIT DATA ====================
print("\nSplitting data into Train/Val/Test...")
X = df.drop(TARGET_COLUMN).to_pandas()
y = df[TARGET_COLUMN].to_pandas()

# 60% Train, 20% Validation, 20% Test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE  # 0.25 * 0.8 = 0.2 of total
)

print(f"Train: {len(X_train)} records")
print(f"Val:   {len(X_val)} records")
print(f"Test:  {len(X_test)} records")

# ==================== SAVE TEST DATA TO MINIO ====================
print(f"\nSaving test data to s3://{OUTPUT_BUCKET}/{TEST_DATA_KEY}")
try:
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    local_test_path = "/tmp/test_data.parquet"
    test_df.to_parquet(local_test_path, index=False)
    s3.upload_file(local_test_path, OUTPUT_BUCKET, TEST_DATA_KEY)
    print(f"Test data saved ({len(test_df)} records).")
except Exception as e:
    print(f"Failed to save test data: {e}")
    sys.exit(1)

# ==================== OPTUNA OBJECTIVE ====================
def objective(trial):
    """
    Optuna objective function.
    Tunes hyperparameters to MAXIMIZE R² on the validation set.
    """
    n_estimators = trial.suggest_int("n_estimators", 5, 10)
    max_depth = trial.suggest_int("max_depth", 4, 10)

    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,  
    )
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)

    return r2

# ==================== RUN OPTUNA STUDY ====================
print(f"\n Running Optuna study ({N_TRIALS} trials)...")
study = optuna.create_study(
    direction="maximize",  
    sampler=TPESampler(seed=RANDOM_STATE), 
    study_name=f"rf_study_{datetime.now().strftime('%Y%m%d')}"
)

study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=False)

print(f"\n Optuna study complete.")
print(f"Best R² (validation): {study.best_value:.4f}")
print(f"Best params: {study.best_trial.params}")

# ==================== TRAIN FINAL MODEL ====================
print("\n Training final model on Train + Validation...")
best_params = study.best_trial.params

X_final = pd.concat([X_train, X_val], axis=0)
y_final = pd.concat([y_train, y_val], axis=0)

final_model = RandomForestRegressor(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
final_model.fit(X_final, y_final)
print("Final model trained.")

# ==================== LOG TO MLFLOW ====================
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run() as run:
    # Log Optuna metadata
    mlflow.log_param("optuna_trials", N_TRIALS)
    mlflow.log_param("optuna_timeout", TIMEOUT)
    mlflow.log_param("best_val_r2", study.best_value)
    mlflow.log_params(best_params)

    mlflow.log_metric("val_r2", study.best_value)

    # Log each trial as nested runs
    for trial in study.trials:
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(trial.params)
            if trial.value is not None:
                mlflow.log_metric("val_r2", trial.value)

    # Register the model 
    signature = infer_signature(X_final, y_final)
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        signature=signature,
        registered_model_name=MODEL_NAME,
    )
    run_id = run.info.run_id
    print(f"Model registered to MLflow. Run ID: {run_id}")

# Save outputs for Argo 
with open("/tmp/run_id.txt", "w") as f:
    f.write(run_id)
with open("/tmp/model_uri.txt", "w") as f:
    f.write(f"runs:/{run_id}/model")

print("Training completed successfully!")

