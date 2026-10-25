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
    python scripts/push_models.py --check              # local validation, no GCS auth needed
    python scripts/push_models.py [--dry-run] [--bucket BUCKET] [--keep N]

    --check     Validate manifest.json and confirm all files exist on disk. Exits 0
                on success, 1 on failure. Does not require GCS credentials.
    --dry-run   Show what would be uploaded without writing.
    --bucket    GCS bucket (default: env GCS_BUCKET or energy-autoresearch-model).
    --keep N    Retain only the N most recent versioned prefixes (default: 3).
                Orphaned prefixes are deleted after the pointer is updated.

Requires (for --dry-run / live push):
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
POINTER_OBJ    = "models/pointer.json"
MANIFEST_FILE  = MODELS_DIR / "manifest.json"
# Matches only versioned prefixes like models/20261024_143022/ — never flat models/*.
_VERSIONED_RE = re.compile(r'^models/\d{8}_\d{6}/$')


def _local_md5_hex(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _gcs_md5_hex(blob) -> str:
    raw = base64.b64decode(blob.md5_hash)
    return raw.hex()


def _list_versioned_prefixes(bucket) -> list[str]:
    """Return all models/<timestamp>/ prefixes, sorted oldest-first.

    Uses delimiter="/" so GCS returns virtual-directory prefixes separately from
    flat blob names.  Only prefixes matching _VERSIONED_RE are returned, which
    guarantees flat models/*.pkl / models/*.json are never selected for deletion.
    """
    iterator = bucket.client.list_blobs(bucket, prefix="models/", delimiter="/")
    prefixes: list[str] = []
    for page in iterator.pages:
        prefixes.extend(page.prefixes)
    return sorted(p for p in prefixes if _VERSIONED_RE.match(p))


def _load_manifest() -> list[Path]:
    """Return the list of local model files to push, as written by train_all.py.

    Aborts if manifest.json is missing (stale models/ dir) or if any listed
    file is not present on disk.
    """
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"{MANIFEST_FILE} not found — run train_all.py first.\n"
            "manifest.json lists exactly the artifacts produced by the last training "
            "run so that stale files from previous runs are never pushed to GCS."
        )
    manifest = json.loads(MANIFEST_FILE.read_text())
    listed   = manifest.get("files", [])
    paths    = [MODELS_DIR / f for f in listed]
    missing  = [p for p in paths if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing[:5])
        raise FileNotFoundError(
            f"manifest.json references {len(missing)} missing file(s): {names}"
            + (" ..." if len(missing) > 5 else "")
        )
    # Include manifest.json itself so GCS carries a record of what was trained.
    return [MANIFEST_FILE] + paths


def main(dry_run: bool, bucket_name: str, keep: int) -> None:
    from google.cloud import storage

    client    = storage.Client()
    bucket    = client.bucket(bucket_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix    = f"models/{timestamp}/"

    files    = _load_manifest()
    uploaded = 0

    print(f"Target prefix: gs://{bucket_name}/{prefix}")
    print(f"Manifest: {len(files) - 1} model files + manifest.json = {len(files)} total")
    for local_path in files:
        rel     = local_path.relative_to(MODELS_DIR)
        gcs_key = prefix + rel.as_posix()
        blob    = bucket.blob(gcs_key)

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


def check() -> None:
    """Validate manifest.json and confirm all files exist on disk. No GCS auth needed."""
    import sys
    try:
        files = _load_manifest()   # raises FileNotFoundError on any problem
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
    manifest = json.loads(MANIFEST_FILE.read_text())
    model_files = files[1:]   # strip manifest.json itself (first entry)
    print(f"manifest.json  trained_at={manifest.get('trained_at', '?')}")
    print(f"Model files:   {len(model_files)} (per manifest)")
    print(f"With manifest: {len(files)} total")
    for p in files:
        size_kb = p.stat().st_size / 1024
        print(f"  OK  {p.name:50s}  {size_kb:8.1f} KB")
    print(f"\nOK — {len(files)} file(s) ready to push.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="Validate manifest and file presence locally (no GCS needed)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--keep", type=int, default=3,
                        help="Versioned prefixes to retain (default 3)")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        main(args.dry_run, args.bucket, args.keep)
