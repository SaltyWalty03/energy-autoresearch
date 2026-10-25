"""
Tests for push_models.py orphan-cleanup logic.

Covers:
  - _VERSIONED_RE applied to prefixes (delimiter="/" listing), not blob names
  - _list_versioned_prefixes filters out flat models/*.pkl and non-timestamp dirs
  - Only the oldest N-keep prefixes are selected for deletion
  - Flat files are never reachable by the deletion path
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

# push_models.py lives in scripts/, not on sys.path by default.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from push_models import _list_versioned_prefixes, _VERSIONED_RE  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_bucket(prefixes: list[str]) -> MagicMock:
    """
    Return a MagicMock bucket whose client.list_blobs() yields one page
    containing the given prefixes (simulates GCS delimiter="/" behaviour).

    Prefixes correspond to virtual subdirectories; flat blob names are NOT in
    page.prefixes — they appear only in the iterated items, which we leave empty.
    This mirrors real GCS: list_blobs(..., delimiter="/") separates blobs (items)
    from subdirectory prefixes in page.prefixes.
    """
    page = MagicMock()
    page.prefixes = prefixes

    iterator = MagicMock()
    iterator.pages = [page]

    bucket = MagicMock()
    bucket.client.list_blobs.return_value = iterator
    return bucket


# ── _VERSIONED_RE unit tests ───────────────────────────────────────────────────

class TestVersionedRe:
    def test_valid_prefix_matches(self):
        assert _VERSIONED_RE.match("models/20261024_143022/") is not None

    def test_missing_trailing_slash_rejected(self):
        assert _VERSIONED_RE.match("models/20261024_143022") is None

    def test_short_date_rejected(self):
        assert _VERSIONED_RE.match("models/2026102_143022/") is None

    def test_short_time_rejected(self):
        assert _VERSIONED_RE.match("models/20261024_14302/") is None

    def test_named_subdirectory_rejected(self):
        assert _VERSIONED_RE.match("models/staging/") is None
        assert _VERSIONED_RE.match("models/backup_old/") is None

    def test_flat_blob_names_rejected(self):
        """Flat blob names must never match — they are not prefixes at all."""
        for name in [
            "models/XLE_model.pkl",
            "models/XLE_meta.json",
            "models/pointer.json",
            "models/manifest.json",
        ]:
            assert _VERSIONED_RE.match(name) is None, f"{name!r} should not match"


# ── _list_versioned_prefixes ───────────────────────────────────────────────────

class TestListVersionedPrefixes:
    def test_all_versioned_prefixes_returned(self):
        prefixes = [
            "models/20261020_100000/",
            "models/20261021_100000/",
            "models/20261022_100000/",
        ]
        bucket = _mock_bucket(prefixes)
        assert _list_versioned_prefixes(bucket) == sorted(prefixes)

    def test_non_versioned_prefixes_excluded(self):
        prefixes = [
            "models/20261020_100000/",
            "models/staging/",
            "models/backup_old/",
            "models/20261021_100000/",
        ]
        bucket = _mock_bucket(prefixes)
        result = _list_versioned_prefixes(bucket)
        assert result == ["models/20261020_100000/", "models/20261021_100000/"]
        assert "models/staging/" not in result
        assert "models/backup_old/" not in result

    def test_result_is_sorted_oldest_first(self):
        # Supply out-of-order to confirm sorting
        prefixes = [
            "models/20261024_100000/",
            "models/20261020_100000/",
            "models/20261022_100000/",
        ]
        bucket = _mock_bucket(prefixes)
        result = _list_versioned_prefixes(bucket)
        assert result == [
            "models/20261020_100000/",
            "models/20261022_100000/",
            "models/20261024_100000/",
        ]

    def test_uses_delimiter_listing(self):
        """
        Confirm the function queries GCS with delimiter="/".
        Flat blob names like XLE_model.pkl are in list_blobs() items, NOT in
        page.prefixes — the regex guard is the second line of defence against
        non-prefix strings that could somehow leak into prefixes.
        """
        bucket = _mock_bucket([])
        _list_versioned_prefixes(bucket)
        bucket.client.list_blobs.assert_called_once()
        _, kwargs = bucket.client.list_blobs.call_args
        assert kwargs.get("delimiter") == "/"

    def test_empty_listing_returns_empty_list(self):
        bucket = _mock_bucket([])
        assert _list_versioned_prefixes(bucket) == []


# ── Orphan cleanup selection logic ────────────────────────────────────────────

class TestOrphanCleanupSelection:
    """
    Simulate the keep logic from main():
        versioned = _list_versioned_prefixes(bucket)
        to_delete = versioned[:-keep] if len(versioned) > keep else []

    The GCS deletion itself is not tested here (requires live GCS); we verify
    that the right set of prefixes is chosen for deletion.
    """

    def _select_orphans(self, prefixes: list[str], keep: int) -> list[str]:
        bucket = _mock_bucket(prefixes)
        versioned = _list_versioned_prefixes(bucket)
        return versioned[:-keep] if len(versioned) > keep else []

    def test_oldest_two_deleted_when_five_kept_three(self):
        prefixes = [
            "models/20261020_100000/",
            "models/20261021_100000/",
            "models/20261022_100000/",
            "models/20261023_100000/",
            "models/20261024_100000/",
        ]
        to_delete = self._select_orphans(prefixes, keep=3)
        assert to_delete == [
            "models/20261020_100000/",
            "models/20261021_100000/",
        ]

    def test_no_deletion_when_exactly_at_keep(self):
        prefixes = [
            "models/20261022_100000/",
            "models/20261023_100000/",
            "models/20261024_100000/",
        ]
        assert self._select_orphans(prefixes, keep=3) == []

    def test_no_deletion_when_below_keep(self):
        prefixes = [
            "models/20261023_100000/",
            "models/20261024_100000/",
        ]
        assert self._select_orphans(prefixes, keep=3) == []

    def test_flat_names_never_in_deletion_set(self):
        """
        Even if flat file names somehow appear as prefixes (impossible in real
        GCS with delimiter="/", but we guard anyway), the regex must filter them.
        """
        prefixes = [
            "models/20261020_100000/",
            "models/20261021_100000/",
            "models/20261022_100000/",
            "models/20261023_100000/",
            "models/staging/",          # not versioned
            "models/20261024_100000/",
        ]
        to_delete = self._select_orphans(prefixes, keep=3)
        assert "models/staging/" not in to_delete
        # Only the two genuinely oldest versioned prefixes are deleted
        assert to_delete == [
            "models/20261020_100000/",
            "models/20261021_100000/",
        ]

    def test_keep_1_deletes_all_but_newest(self):
        prefixes = [
            "models/20261020_100000/",
            "models/20261021_100000/",
            "models/20261022_100000/",
        ]
        to_delete = self._select_orphans(prefixes, keep=1)
        assert to_delete == [
            "models/20261020_100000/",
            "models/20261021_100000/",
        ]
