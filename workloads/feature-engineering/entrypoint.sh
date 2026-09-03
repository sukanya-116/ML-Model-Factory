#!/bin/bash

set -e

TASK=${TASK:-"feature_engineering"}

echo "Running task: $TASK"

case $TASK in
    "validation")
        python /app/validation.py   
    ;;
    "feature_engineering")
        python /app/featurizer.py
    ;;
  *)
    echo "Unknown task: $TASK"
    echo "Available tasks: validation, feature_engineering"
    exit 1
    ;;
esac

echo "Task $TASK completed successfully."

