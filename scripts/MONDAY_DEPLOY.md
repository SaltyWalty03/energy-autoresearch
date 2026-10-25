# Monday Deploy Checklist — energy-autoresearch

Run each step in order. Stop and investigate if any step fails.
Confirm with `!` before any step that submits real orders.

---

## a. Branch and working-tree state

```bash
git branch                          # must be on infra-hardening
git status                          # must be clean (or only expected untracked)
git log --oneline -5                # confirm latest infra-hardening commit is present
```

---

## b. Sync check — cloud function vs repo

```bash
source .venv/Scripts/activate
python scripts/sync_cloud_function.py --check
```

Exits 0 if `cloud_function/` matches `src/` + shared modules. Fix any drift before continuing.

---

## c. Local dry-run — capture expected orders

```bash
DRY_RUN=true python cloud_function/main.py --dry-run --skip-hours-check 2>&1 | tee /tmp/dryrun_$(date +%Y%m%d).json
```

**Expected orders = the output from this step.** Read it carefully:
- Review each asset's `action` field (buy/sell/skip/flat).
- Verify `regime`, `wf_sharpe`, and `qty` look plausible given current positions.
- If any asset shows `bias_error` or `close_failed`, stop and investigate.

There is no hardcoded order table. The dry-run output is the ground truth.

---

## d. Regime stability review (informational, do not act on)

Current training results as of 2026-10-24:

| Asset | Regimes | Flagged (< 100 days) |
|-------|---------|----------------------|
| XLE   | 3       | R1: 54 rows          |
| XLB   | 11      | R1: 96, R2: 54, R7: 90, R9: 67, R10: 97 (5 regimes) |
| XOP   | 14      | R2: 70, R5: 65, R6: 72, R8: 81, R9: 96, R13: 99 (6 regimes) |

12 of 28 regimes have fewer than 100 training days. These are live but may have unreliable
walk-forward Sharpes. Do not retrain or adjust parameters at this step — flag for next
training cycle.

---

## e. Verify manifest before push

`train_all.py` writes `models/manifest.json` listing every artifact it produced.
`push_models.py` reads this manifest and uploads exactly those files — stale artifacts
from previous training runs are never pushed regardless of what else is in `models/`.

```bash
python -c "import json; m=json.load(open('models/manifest.json')); print(m['file_count'], 'files,', m['trained_at'])"
```

Expected output: `65 files, 2026-10-24T...` (XLE 9 + XLB 25 + XOP 31).
If `manifest.json` is missing, re-run `python train_all.py` first.

Active file count breakdown:
- XLE:  3 regimes → 3×2 + model + meta + regime.pkl = **9 files**
- XLB: 11 regimes → 11×2 + model + meta + regime.pkl = **25 files**
- XOP: 14 regimes → 14×2 + model + meta + regime.pkl = **31 files**
- **Total: 65 files** (plus manifest.json = 66 pushed to GCS)

---

## f. Push models to GCS

```bash
python scripts/push_models.py --dry-run              # preview; confirm 66 files shown (65 + manifest)
python scripts/push_models.py --keep 3               # publish; writes pointer.json
```

Verify output:
- "66 file(s) would be uploaded." (dry-run) / "66 file(s) uploaded." (live)
- "Pointer updated: models/pointer.json → models/YYYYMMDD_HHMMSS/"
- Orphan cleanup log (keeps 3 most recent versioned prefixes; never deletes flat models/).

---

## g. Deploy cloud function

```bash
bash scripts/deploy.sh
```

`deploy.sh` runs `sync_cloud_function.py --check` first and aborts if files are mismatched.
Confirm gcloud output shows revision deployed and traffic 100% to new revision.

---

## h. Post-deploy smoke test

```bash
gcloud run services describe run-daily --region us-central1 --format="value(status.url)"
# then hit the URL once to confirm HTTP 200 (outside market hours returns {"status":"closed"})
curl -s "$(gcloud run services describe run-daily --region us-central1 \
    --format='value(status.url)')" | python3 -m json.tool
```

Expected: `{"status": "closed", "message": "Outside market hours ..."}` — confirms the
function is alive and the market-hours guard works.

---

## i. Apply monitoring alerts

```bash
bash monitoring/APPLY.sh
```

Creates (or reuses) the email channel, log-based metric `decision_count`, and three alert
policies: 5xx errors, scheduler failures, heartbeat (no decision in 8 h). Check gcloud
output for "Created" or "Already exists" — both are fine.

---

## j. Confirm first live heartbeat (next trading day)

After market open (~9:30 AM ET), check Cloud Logging within 15 minutes:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="run-daily"
   AND (jsonPayload.message=~"^DECISION" OR jsonPayload.message=~"^NO TRADE")' \
  --project=energy-autoresearch --limit=5 --format=json
```

Confirms the scheduler is firing and the function is reaching the decision stage.
The heartbeat alert fires if this is absent for a full 8-hour trading window.

---

## Rollback

If step h or j shows errors, roll back immediately:

**1. Roll back the cloud function:**
```bash
# List recent revisions
gcloud run revisions list --service run-daily --region us-central1

# Route 100% traffic to the previous revision
gcloud run services update-traffic run-daily --region us-central1 \
    --to-revisions=<PREV_REVISION>=100
```

**2. Roll back GCS models (point to previous versioned prefix):**
```bash
# List versioned prefixes in GCS
gsutil ls gs://energy-autoresearch-model/models/ | grep -E 'models/[0-9]{8}_[0-9]{6}/'

# Write pointer.json to the last-known-good prefix
python3 -c "
import json
from google.cloud import storage
from datetime import datetime, timezone
client = storage.Client()
prefix = 'models/YYYYMMDD_HHMMSS/'   # replace with the rollback target
client.bucket('energy-autoresearch-model').blob('models/pointer.json').upload_from_string(
    json.dumps({'prefix': prefix, 'uploaded_at': datetime.now(timezone.utc).isoformat(),
                'note': 'manual rollback'}),
    content_type='application/json')
print('Pointer rolled back to', prefix)
"
```

**3. Verify rollback with dry-run:**
```bash
DRY_RUN=true python cloud_function/main.py --dry-run --skip-hours-check
```

Confirm the expected-orders output returns to pre-deploy baseline before the next trading session.
