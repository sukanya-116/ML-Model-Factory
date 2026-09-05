#!/usr/bin/env python3
"""
Model Training Script for Mobile Sales Dataset
"""
import os
import sys
import boto3
import polars as pl
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json, io
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET = os.getenv("OUTPUT_BUCKET", "processed-features")
KEY = os.getenv("FEATURES_KEY", "")  
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-mlflow.mldata.svc.cluster.local:5000")

def train_model():
    print(f"🚀 Starting Model Training from s3://{BUCKET}/{KEY}")
    
    # 1. Initialize S3 client
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    
    # 2. Load features from MinIO
    try:
        
        obj = s3.get_object(Bucket=BUCKET, Key=KEY)
        data = io.BytesIO(obj['Body'].read())
        df = pl.read_parquet(data)
        print(f"Loaded {len(df)} records from MinIO.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)
    
    # 3. Prepare X (features) and y (target)
    target = "Quantity Sold"
    X = df.drop(target).to_pandas()
    y = df[target].to_pandas()
    
    # 4. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 5. Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # 6. Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")
    
    # 7. Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("mobile_sales_prediction")
    
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("features", list(X.columns))
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        mlflow.sklearn.log_model(model, "model")

        print(f"Logged model to MLflow run: {run.info.run_id}")
    
    # 8. Save the model URI to a file for Argo to capture
    with open("/tmp/model_uri.txt", "w") as f:
        f.write(f"runs:/{run.info.run_id}/model")
    
    print("Model training completed successfully!")

if __name__ == "__main__":
    train_model()