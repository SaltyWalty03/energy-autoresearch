#!/usr/bin/env bash
# Apply all Cloud Monitoring alerts for energy-autoresearch.
# Run once; re-run to update (policies are idempotent by display name if you
# delete the old one first with: gcloud monitoring policies delete <NAME>).
set -euo pipefail

PROJECT=energy-autoresearch

# ── 0. Find or create an email notification channel ──────────────────────────
CHANNEL=$(gcloud monitoring channels list \
    --project="$PROJECT" \
    --filter='type=email AND displayName="energy-autoresearch alerts"' \
    --format='value(name)' | head -1)

if [ -z "$CHANNEL" ]; then
    echo "Creating email notification channel..."
    CHANNEL=$(gcloud monitoring channels create \
        --project="$PROJECT" \
        --display-name="energy-autoresearch alerts" \
        --type=email \
        --channel-labels=email_address=w.p.brookby@gmail.com \
        --format='value(name)')
    echo "Created: $CHANNEL"
else
    echo "Using existing channel: $CHANNEL"
fi

# ── 1. Create log-based metric (heartbeat counter) ───────────────────────────
# Only create if it does not already exist.
if ! gcloud logging metrics describe decision_count --project="$PROJECT" &>/dev/null; then
    echo "Creating log-based metric decision_count..."
    gcloud logging metrics create decision_count \
        --project="$PROJECT" \
        --description="DECISION/NO TRADE lines from run-daily" \
        --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="run-daily" AND (jsonPayload.message=~"^DECISION" OR jsonPayload.message=~"^NO TRADE")'
else
    echo "Log metric decision_count already exists."
fi

DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 2. Apply alert policies from JSON ────────────────────────────────────────
for policy_file in \
    "$DIR/alert_5xx.json" \
    "$DIR/alert_scheduler.json" \
    "$DIR/alert_heartbeat.json"
do
    echo "Applying $policy_file ..."
    # Inject the notification channel into the policy JSON
    tmp=$(mktemp /tmp/policy_XXXXXX.json)
    python3 -c "
import json, sys
data = json.load(open('$policy_file'))
data['notificationChannels'] = ['$CHANNEL']
json.dump(data, open('$tmp','w'), indent=2)
"
    gcloud monitoring policies create \
        --project="$PROJECT" \
        --policy-from-file="$tmp"
    rm "$tmp"
done

echo "==> All alerts applied."
