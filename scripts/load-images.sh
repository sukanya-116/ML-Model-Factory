#!/bin/bash
# scripts/load-images.sh
#
# Loads all ML Factory images into the Kind cluster.
# Run this script from the project root: ./scripts/load-images.sh

set -e

# ============================================================
# Configuration
# ============================================================
CLUSTER_NAME="${CLUSTER_NAME:-mlfactory}"   
TAG="${TAG:-v1}"
REGISTRY="${REGISTRY:-}"                     

echo "Loading images into Kind cluster: ${CLUSTER_NAME}"
echo "Tag: ${TAG}"
echo "Registry: ${REGISTRY:-<local>}"
echo "============================================================"

# ============================================================
# Check that the cluster exists
# ============================================================
if ! kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
  echo "Kind cluster '${CLUSTER_NAME}' not found."
  echo "Create it with: kind create cluster --name ${CLUSTER_NAME}"
  exit 1
fi

# ============================================================
# Load each image
# ============================================================
WORKLOADS=(
  "data-validation"
  "feature-engineering"
  "model-training"
  "model-evaluation"
  "model-promotion"
)

for workload in "${WORKLOADS[@]}"; do
  IMAGE_NAME="${REGISTRY}${workload}:${TAG}"

  # Check if the image exists locally
  if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "Image ${IMAGE_NAME} not found locally. Skipping."
    echo "Run ./scripts/build-all.sh first."
    continue
  fi

  echo ""
  echo "Loading ${IMAGE_NAME}..."
  kind load docker-image "${IMAGE_NAME}" --name "${CLUSTER_NAME}"
  echo "Loaded ${IMAGE_NAME}"
done

echo ""
echo "============================================================"
echo "All images loaded into Kind!"
echo ""
echo "Verify:"
echo "  docker exec -it ${CLUSTER_NAME}-control-plane crictl images | grep -E 'data-validation|feature-engineering|model-'"
echo "============================================================"