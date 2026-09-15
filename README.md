# ML Model Factory

An end-to-end, Kubernetes-native **MLOps platform** that automates the complete machine learning lifecycle: **data validation → feature engineering → model training → evaluation → promotion → serving → monitoring**.

[![Python](https://img.shields.io/badge/Python-3.10-0D47A1?style=for-the-badge&labelColor=2E7D32&logo=python&logoColor=white)](https://www.python.org/)
[![Orchestration](https://img.shields.io/badge/Orchestration-Argo_Workflows-0D47A1?style=for-the-badge&labelColor=2E7D32&logo=argo&logoColor=white)](https://argoproj.github.io/workflows/)
[![Serving](https://img.shields.io/badge/Serving-KServe-0D47A1?style=for-the-badge&labelColor=2E7D32&logo=kubernetes&logoColor=white)](https://kserve.github.io/website/)
[![Service_Mesh](https://img.shields.io/badge/Service_Mesh-Istio-0D47A1?style=for-the-badge&labelColor=2E7D32&logo=istio&logoColor=white)](https://istio.io/)
[![Serverless](https://img.shields.io/badge/Serverless-Knative-0D47A1?style=for-the-badge&labelColor=2E7D32&logo=knative&logoColor=white)](https://knative.dev/)
[![Storage](https://img.shields.io/badge/Storage-MinIO-0D47A1?style=for-the-badge&labelColor=2E7D32&logo=minio&logoColor=white)](https://min.io/)
[![Registry](https://img.shields.io/badge/Registry-MLflow-0D47A1?style=for-the-badge&labelColor=2E7D32&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![HPO](https://img.shields.io/badge/HPO-Optuna-0D47A1?style=for-the-badge&labelColor=2E7D32&logo=optuna&logoColor=white)](https://optuna.org/)
[![Metrics](https://img.shields.io/badge/Metrics-Prometheus-0D47A1?style=for-the-badge&labelColor=2E7D32&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Dashboards](https://img.shields.io/badge/Dashboards-Grafana-0D47A1?style=for-the-badge&labelColor=2E7D32&logo=grafana&logoColor=white)](https://grafana.com/)
[![Containerization](https://img.shields.io/badge/Containerization-Docker-0D47A1?style=for-the-badge&labelColor=2E7D32&logo=docker&logoColor=white)](https://www.docker.com/)
[![Cloud](https://img.shields.io/badge/Cloud-Kubernetes-0D47A1?style=for-the-badge&labelColor=2E7D32&logo=kubernetes&logoColor=white)](https://kubernetes.io/)

### What It Does

Given a raw dataset (e.g., mobile sales data), the factory:

1. **Validates** the data against a quality contract.
2. **Engineers** features using Polars for speed.
3. **Trains** a model with automated hyperparameter tuning (Optuna).
4. **Evaluates** the new model against the current production champion.
5. **Promotes** the model only if it improves business metrics.
6. **Serves** the model via KServe with zero-downtime rolling updates.
7. **Monitors** the entire stack with Prometheus + Grafana.

### Business Use Case (Example)

**Mobile Sales Prediction**: Predict `Quantity Sold` for a product on a given dispatch date, enabling inventory optimization and demand planning.


## 🏗️ Architecture

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    USER / DATA SCIENTIST                         │
└─────────────┬────────────────────────────────┬───────────────────┘
              │                                │
              │ 1. Submit pipeline             │ 2. Send prediction
              ▼                                ▼
┌──────────────────────────┐    ┌──────────────────────────────────┐
│  ARGO WORKFLOWS          │    │  KServe + Istio + Knative        │
│  (Orchestration)         │    │  (Serving)                       │
│                          │    │                                  │
│  ┌────────────────────┐  │    │  ┌────────────────────────────┐  │
│  │ 1. Data Validation │  │    │  │ InferenceService           │  │
│  └─────────┬──────────┘  │    │  └─────────┬──────────────────┘  │
│            ▼             │    │            ▼                     │
│  ┌────────────────────┐  │    │  ┌────────────────────────────┐  │
│  │ 2. Feature Eng.    │  │    │  │ Revision (Pod)             │  │
│  └─────────┬──────────┘  │    │  │ + istio-proxy (metrics)    │  │
│            ▼             │    │  └────────────────────────────┘  │
│  ┌────────────────────┐  │    └──────────────────────────────────┘
│  │ 3. Model Training  │  │                    │
│  │    (Optuna HPO)    │  │                    │ metrics
│  └─────────┬──────────┘  │                    ▼
│            ▼             │    ┌──────────────────────────────────┐
│  ┌────────────────────┐  │    │  PROMETHEUS + GRAFANA            │
│  │ 4. Model Evaluation│  │    │  (Observability)                 │
│  └─────────┬──────────┘  │    └──────────────────────────────────┘
│            ▼             │
│  ┌────────────────────┐  │
│  │ 5. Model Promotion │──┼──▶ Patches InferenceService
│  └────────────────────┘  │
└──────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│  MLFLOW + MINIO + POSTGRESQL                                     │
│  (Model Registry + Artifact Storage)                             │
└──────────────────────────────────────────────────────────────────┘
```

### Namespaces

| Namespace | Purpose | Components |
| :--- | :--- | :--- |
| `mldata` | Data & ML storage | MinIO, PostgreSQL, MLflow |
| `argo` | Pipeline orchestration | Argo Workflows |
| `istio-system` | Service mesh | Istio control plane, Prometheus, Grafana |
| `knative-serving` | Serverless runtime | Knative controller, autoscaler, activator |
| `kserve` | Model serving controller | KServe controller |
| `ml-serving` | Deployed models | InferenceService, revisions, pods |

## 🛠️ Tech Stack

### Core Infrastructure

| Component | Version | Purpose |
| :--- | :--- | :--- |
| **Kubernetes** | v1.27+ | Container orchestration |
| **Kind** | Latest | Local K8s cluster |
| **MinIO** | Latest | S3-compatible object storage |
| **PostgreSQL** | 15 | Metadata store for MLflow |

### MLOps Platform

| Component | Version | Purpose |
| :--- | :--- | :--- |
| **MLflow** | 2.11.0 | Experiment tracking + model registry |
| **Argo Workflows** | v3.5.9 | Pipeline orchestration |
| **KServe** | v0.12.0 | Model serving |
| **Knative Serving** | v1.12.0 | Serverless autoscaling |
| **Istio** | 1.18.0 | Service mesh + traffic routing |

### Observability

| Component | Version | Purpose |
| :--- | :--- | :--- |
| **Prometheus** | Istio addon | Metrics collection |
| **Grafana** | Istio addon | Visualization |

### Python Stack

| Package | Purpose |
| :--- | :--- |
| **Polars** | Fast DataFrame library |
| **Boto3** | S3/MinIO access |
| **Scikit-learn** | Model training |
| **Optuna** | Hyperparameter tuning |
| **MLflow SDK** | Experiment tracking |

## 📁 Repository Structure

```
ml-factory/
├── README.md
├── pyproject.toml                    # Root dev environment
├── uv.lock
│
├── notebooks/                         # Exploration & testing
│   ├── 1.data_validation.ipynb
│   ├── 2.feature_engineering.ipynb
│   ├── 3.model_training.ipynb
│   └── 4.evaluate.ipynb
│   └── 5.promote.ipynb
│
├── infrastructure/                    # Platform components
│   ├── minio/
│   │   └── values.yaml
│   ├── mlflow/
│   |   └── values.yaml
|   └── postgres/
|       └── values.yaml
│
├── workloads/                         # Pipeline components
│   ├── data-validation/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── validation.py
│   │
│   ├── feature-engineering/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── featurizer.py
│   │
│   ├── model-training/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── train.py
│   │
│   ├── model-evaluation/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── evaluate.py
│   │
│   └── model-promotion/
│       ├── Dockerfile
│       ├── pyproject.toml
│       └── promote.py
│
├── argo-workflows/                    # Pipeline definitions
│   └── ml-pipeline.yaml
│
├── serving/                           # Model serving configs
│   └── inference-service.yaml
│
├── grafana/                           # Dashboards 
│   └── dashboard.json
│
└── scripts/                           # Utility scripts
    ├── build-all.sh
    └── load-images.sh


```

## ✅ Prerequisites

### Local Machine

- **Docker** (v20.10+)
- **kubectl** (v1.27+)
- **Kind** (v0.20+) or Minikube
- **Helm** (v3.12+)
- **uv** (Python package manager) – [Install](https://docs.astral.sh/uv/)
- **Python** 3.10

## 🚀 Installation

### Step 1: Create the Cluster

```bash
kind create cluster --name mlfactory --config - <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30000
        hostPort: 30000
      - containerPort: 30001
        hostPort: 30001
EOF
```

### Step 2: Install Core Infrastructure

```bash
# 2.1 MinIO (S3 storage)
helm repo add minio https://charts.min.io/
helm install mlminio minio/minio -n mldata --create-namespace \
  -f infrastructure/minio/values.yaml

# 2.2 PostgreSQL (metadata store)
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install mlflow-postgres bitnami/postgresql -n mldata \
  -f infrastructure/postgresql/values.yaml

# 2.3 MLflow (experiment tracking)
helm install mlflow oci://ghcr.io/mlflow/charts/mlflow \
  --version 0.1.0 -n mldata -f infrastructure/mlflow/values.yaml
```

### Step 3: Install Orchestration

```bash
# 3.1 Argo Workflows
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.9/install.yaml

# 3.2 Patch Argo server (disable TLS + auth for dev)
kubectl patch deployment argo-server -n argo --type='json' \
  -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--secure=false"}]'
kubectl patch deployment argo-server -n argo --type='json' \
  -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--auth-mode=server"}]'
```

### Step 4: Install Serving Stack

```bash
# 4.1 Cert-Manager (for KServe webhook)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.1/cert-manager.yaml

# 4.2 Knative Serving
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.12.0/serving-crds.yaml
kubectl apply -f https://github.com/knative/serving/releases/download/knative-v1.12.0/serving-core.yaml

# 4.3 Istio
curl -L https://istio.io/downloadIstio | ISTIO_VERSION=1.18.0 sh -
cd istio-1.18.0 && export PATH=$PWD/bin:$PATH
istioctl install --set profile=demo -y
cd ..

# 4.4 Knative Istio integration
kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.12.0/net-istio.yaml

# 4.5 KServe (with RBAC proxy fix)
curl -sL https://github.com/kserve/kserve/releases/download/v0.12.0/kserve.yaml | \
  sed 's|gcr.io/kubebuilder/kube-rbac-proxy:v0.13.1|quay.io/brancz/kube-rbac-proxy:v0.18.1|g' | \
  kubectl apply -f -
```

### Step 5: Install Observability

```bash
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.18/samples/addons/prometheus.yaml
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.18/samples/addons/grafana.yaml
```

### Step 6: Enable Istio Sidecar Injection

```bash
kubectl label namespace ml-serving istio-injection=enabled
kubectl label namespace knative-serving istio-injection=enabled
```

### Step 7: Build & Load Container Images

```bash
./scripts/build-all.sh
./scripts/load-images.sh
```

## 🎬 Running the Pipeline

### Submit the Pipeline

```bash
kubectl create -f argo-workflows/ml-pipeline.yaml -n argo
```

### Monitor the Pipeline

```bash
# List all workflows
kubectl get workflows -n argo

# View logs of a specific step
kubectl logs -n argo -l workflows.argoproj.io/workflow=<workflow-name> -c main

# Open the Argo UI
kubectl port-forward -n argo svc/argo-server 2746:2746
# Open http://localhost:2746
```

### Pipeline Stages

| Stage | Description | Output |
| :--- | :--- | :--- |
| `data-validation` | Validate raw CSV schema and quality. | `validation_report.json` |
| `feature-engineering` | Transform data into ML features. | `features_*.parquet` |
| `model-training` | Train with Optuna HPO. | Registered model in MLflow |
| `model-evaluation` | Compare with champion on test set. | `decision.json` |
| `model-promotion` | Patch InferenceService with champion. | Updated KServe deployment |

## 🚀 Model Serving

### Deploy the InferenceService

```bash
kubectl apply -f serving/inference-service.yaml
```

### Check Status

```bash
kubectl get inferenceservice mobile-sales-predictor -n ml-serving
kubectl get pods -n ml-serving
```

### Test Predictions

```bash
# Port-forward the local gateway
kubectl port-forward -n istio-system svc/knative-local-gateway 8080:80 &

# Get the InferenceService host
HOST=$(kubectl get inferenceservice mobile-sales-predictor -n ml-serving -o jsonpath='{.status.url}' | sed 's|http://||')

# Send a prediction
curl -X POST \
  -H "Host: $HOST" \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records": [{"Price": 78570.0, "days_to_sell": 1, "dispatch_year": 2023, "dispatch_month": 8, "dispatch_day_of_week": 3, "spec_length": 45, "brand_code": 5, "region_code": 2, "ram_code": 3, "rom_code": 2, "avg_price_per_brand": 82345.6, "avg_qty_per_region": 5.2}]}' \
  http://localhost:8080/invocations
```

**Expected Response:**
```json
{"predictions": [5.12]}
```
## 📊 Observability

### Access Grafana

```bash
kubectl port-forward -n istio-system svc/grafana 3000:3000
# Open http://localhost:3000 (admin/admin)
```

### Import Dashboard from grafana/dashboard.json

### Key Metrics

| Metric | PromQL | Purpose |
| :--- | :--- | :--- |
| Request Rate | `sum(rate(istio_requests_total{destination_service=~".*mobile-sales.*"}[5m]))` | Predictions per second |
| P95 Latency | `histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket{...}[5m])) by (le))` | Model response time |
| Error Rate | `sum(rate(istio_requests_total{response_code=~"5.."}[5m]))` | 5xx errors |
| Argo Success Rate | `sum(rate(argo_workflows_count{status="Succeeded"}[1h])) / sum(rate(argo_workflows_count[1h]))` | Pipeline health |

---

### Useful Commands

```bash
# Check InferenceService status
kubectl describe inferenceservice mobile-sales-predictor -n ml-serving

# Check KServe controller logs
kubectl logs -n kserve deployment/kserve-controller-manager

# Check Istio proxy logs
kubectl logs -n ml-serving <pod-name> -c istio-proxy

# Check MLflow server logs
kubectl logs -n mldata deployment/mlflow-mlflow
```

## 🤝 Contributing

This is a learning/reference project. Contributions are welcome:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

---