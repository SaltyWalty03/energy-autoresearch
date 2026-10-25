#!/usr/bin/env python3
"""
Upload model artifacts from local models/ to a versioned GCS prefix, then
update the pointer so the cloud function picks up the new version.

Layout in GCS after upload:
    models/<timestamp>/XLE_model.pkl
    models/<timestamp>/XLE_meta.json
    ...
    models/pointer.json  →  {"prefix": "models/20261024_143022/", "uploaded_at": "..."}

The cloud function reads pointer.json first; if missing it falls back to models/.

Usage:
    python scripts/push_models.py [--dry-run] [--bucket BUCKET] [--keep N]

    --dry-run   Show what would be uploaded without writing.
    --bucket    GCS bucket (default: env GCS_BUCKET or energy-autoresearch-model).
    --keep N    Retain only the N most recent versioned prefixes (default: 3).
                Orphaned prefixes are deleted after the pointer is updated.

Requires:
    google-cloud-storage  (pip install google-cloud-storage)
    GCP credentials in environment (gcloud auth application-default login)
"""
import argparse
import base64
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
MODELS_DIR  = ROOT / "models"
DEFAULT_BUCKET = os.environ.get("GCS_BUCKET", "energy-autoresearch-model")
POINTER_OBJ = "models/pointer.json"
# Matches only versioned prefixes like models/20261024_143022/ — never flat models/*.
_VERSIONED_RE = re.compile(r'^models/\d{8}_\d{6}/$')


def _local_md5_hex(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _gcs_md5_hex(blob) -> str:
    raw = base64.b64decode(blob.md5_hash)
    return raw.hex()


def _list_versioned_prefixes(bucket) -> list[str]:
    """Return all models/<timestamp>/ prefixes, sorted oldest-first."""
    blobs = list(bucket.list_blobs(prefix="models/", delimiter="/"))
    # list_blobs with delimiter populates bucket.list_blobs(...).prefixes
    # Re-call to get prefixes
    iterator = bucket.client.list_blobs(bucket, prefix="models/", delimiter="/")
    _ = list(iterator)  # exhaust so .pages populates prefixes
    prefixes = []
    for page in iterator.pages:
        prefixes.extend(page.prefixes)
    # Accept only models/YYYYMMDD_HHMMSS/ — never flat models/*.pkl or models/*.json.
    return sorted(p for p in prefixes if _VERSIONED_RE.match(p))


def main(dry_run: bool, bucket_name: str, keep: int) -> None:
    from google.cloud import storage

    client    = storage.Client()
    bucket    = client.bucket(bucket_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix    = f"models/{timestamp}/"

    files = sorted(MODELS_DIR.rglob("*"))
    uploaded = skipped = 0

    print(f"Target prefix: gs://{bucket_name}/{prefix}")
    for local_path in files:
        if not local_path.is_file():
            continue
        rel      = local_path.relative_to(MODELS_DIR)
        gcs_key  = prefix + rel.as_posix()
        blob     = bucket.blob(gcs_key)
        local_md5 = _local_md5_hex(local_path)

        if not dry_run:
            blob.upload_from_filename(str(local_path))
            uploaded += 1
            print(f"  UPLOAD  {gcs_key}")
        else:
            uploaded += 1
            print(f"  WOULD UPLOAD  {gcs_key}")

    # Write pointer
    pointer_data = json.dumps({
        "prefix": prefix,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "file_count": uploaded,
    })
    if not dry_run:
        bucket.blob(POINTER_OBJ).upload_from_string(pointer_data,
                                                    content_type="application/json")
        print(f"\nPointer updated: {POINTER_OBJ} → {prefix}")
    else:
        print(f"\nWould write pointer: {POINTER_OBJ} → {prefix}")

    # Orphan cleanup: keep only N most recent versioned prefixes
    if not dry_run and keep > 0:
        try:
            versioned = _list_versioned_prefixes(bucket)
            to_delete = versioned[:-keep] if len(versioned) > keep else []
            for old_prefix in to_delete:
                old_blobs = list(bucket.list_blobs(prefix=old_prefix))
                for b in old_blobs:
                    b.delete()
                print(f"  DELETED orphan prefix: {old_prefix} ({len(old_blobs)} objects)")
        except Exception as exc:
            print(f"  WARNING: orphan cleanup failed (non-fatal): {exc}")

    print(f"\n{uploaded} file(s) {'would be ' if dry_run else ''}uploaded.")
    if dry_run:
        print("Re-run without --dry-run to publish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--keep", type=int, default=3,
                        help="Versioned prefixes to retain (default 3)")
    args = parser.parse_args()
    main(args.dry_run, args.bucket, args.keep)
