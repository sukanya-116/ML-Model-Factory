#!/bin/bash
# scripts/build-all.sh
#
# Builds all Docker images for the ML Factory workloads.
# Run this script from the project root: ./scripts/build-all.sh

set -e

# ============================================================
# Configuration
# ============================================================
REGISTRY="${REGISTRY:-}"          # Optional: e.g., "localhost:5000/"
TAG="${TAG:-v1}"                  # Default tag
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo " Building all ML Factory images"
echo " Project root: ${PROJECT_ROOT}"
echo " Tag: ${TAG}"
echo " Registry: ${REGISTRY:-<local>}"
echo "============================================================"

# ============================================================
# Build each workload
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
  echo ""
  echo "Building ${IMAGE_NAME}..."
  docker build \
    -t "${IMAGE_NAME}" \
    -f "${PROJECT_ROOT}/workloads/${workload}/Dockerfile" \
    "${PROJECT_ROOT}/workloads/${workload}/"
  echo "Built ${IMAGE_NAME}"
done

echo ""
echo "============================================================"
echo " All images built successfully!"
echo ""
echo "Next step: load them into your Kind cluster:"
echo "  ./scripts/load-images.sh"
echo "============================================================"