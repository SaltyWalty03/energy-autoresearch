#!/usr/bin/env bash
# Deploy run-daily Cloud Run service.
# Aborts if shared module copies are out of sync with the project root.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Syncing shared modules..."
python scripts/sync_cloud_function.py
python scripts/sync_cloud_function.py --check   # abort if still mismatched

echo "==> Deploying to Cloud Run..."
gcloud run deploy run-daily \
    --source cloud_function/ \
    --region us-central1 \
    --set-secrets ALPACA_API_KEY=ALPACA_API_KEY:latest,ALPACA_SECRET_KEY=ALPACA_SECRET_KEY:latest \
    --project energy-autoresearch

echo "==> Deploy complete."
