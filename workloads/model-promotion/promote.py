import json
import os
import sys
import mlflow
import kubernetes
from kubernetes.client.rest import ApiException
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME", "mobile-sales-predictor")
ALIAS_CHAMPION = os.getenv("ALIAS_CHAMPION", "champion")
NAMESPACE = os.getenv("KSERVE_NAMESPACE", "ml-serving")
SERVICE_NAME = os.getenv("KSERVE_SERVICE_NAME", "mobile-sales-predictor")

# ==================== READ DECISION ====================
with open("/tmp/decision.json", "r") as f:
    decision = json.load(f)

if not decision["promote"]:
    print(f"Promotion not approved. Reason: {decision['reason']}")
    print("Skipping KServe patch.")
    sys.exit(0)

print(f"Promotion approved. Reason: {decision['reason']}")

# ==================== FETCH CHAMPION URI ====================
mlflow.set_tracking_uri(MLFLOW_URI)
client = mlflow.tracking.MlflowClient()

try:
    champion_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_CHAMPION)
    model_uri = f"runs:/{champion_version.run_id}/model"
    print(f"Champion: version={champion_version.version}, uri={model_uri}")
except Exception as e:
    print(f"Failed to fetch champion alias: {e}")
    sys.exit(1)

# ==================== PATCH KSERVE ====================
kubernetes.config.load_incluster_config()
api = kubernetes.client.CustomObjectsApi()
group = "serving.kserve.io"
version = "v1beta1"
plural = "inferenceservices"

try:
    # Get the current InferenceService
    isvc = api.get_namespaced_custom_object(group, version, NAMESPACE, plural, SERVICE_NAME)

    # Update the model URI
    containers = isvc["spec"]["predictor"]["containers"]
    for container in containers:
        for env in container.get("env", []):
            if env["name"] == "MLFLOW_S3_ARTIFACT_URI":
                env["value"] = model_uri
                break

    # Apply the patch
    api.replace_namespaced_custom_object(group, version, NAMESPACE, plural, SERVICE_NAME, isvc)
    print(f"InferenceService '{SERVICE_NAME}' patched to serve '{model_uri}'.")
except ApiException as e:
    print(f"Failed to patch InferenceService: {e}")
    sys.exit(1)