#!/usr/bin/env python3
"""
Sync shared source files from the project root into cloud_function/ and deploy/.

MUST be run before every deploy. The deploy wrapper (scripts/deploy.sh) calls
this automatically and aborts if any checksum differs after the copy.

Usage:
    python scripts/sync_cloud_function.py [--dry-run] [--check]

    --dry-run   Show what would change without writing.
    --check     Verify only — exit 1 if any destination is out of sync.
                Used by scripts/deploy.sh to abort on mismatch.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COPIES = [
    # (source relative to ROOT, list of destinations relative to ROOT)
    ("model.py",        ["cloud_function/model.py", "deploy/model.py"]),
    ("src/regime.py",   ["cloud_function/regime.py",
                         "cloud_function/src/regime.py",
                         "deploy/src/regime.py"]),
    ("src/universe.py", ["cloud_function/src/universe.py",
                         "deploy/src/universe.py"]),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(dry_run: bool, check_only: bool) -> int:
    changed = 0
    for src_rel, dst_rels in COPIES:
        src = ROOT / src_rel
        if not src.exists():
            print(f"  SKIP  {src_rel} — source not found", file=sys.stderr)
            continue
        src_hash = _sha256(src)
        for dst_rel in dst_rels:
            dst = ROOT / dst_rel
            if dst.exists() and _sha256(dst) == src_hash:
                print(f"  ok    {dst_rel}")
                continue
            if check_only:
                print(f"  MISMATCH  {dst_rel}  (run sync_cloud_function.py to fix)", file=sys.stderr)
            else:
                print(f"  COPY  {src_rel} → {dst_rel}")
            changed += 1
            if not dry_run and not check_only:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    if check_only:
        if changed:
            print(f"\nABORT: {changed} destination(s) out of sync — run sync before deploy.",
                  file=sys.stderr)
            return 1
        print("\nAll destinations in sync.")
        return 0

    if dry_run:
        print(f"\nDry run — {changed} file(s) would be updated.")
    else:
        print(f"\n{changed} file(s) updated.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="Verify only, exit 1 if out of sync")
    args = parser.parse_args()
    sys.exit(main(args.dry_run, args.check))
