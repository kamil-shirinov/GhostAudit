"""Shared setup for the offline unit tests.

Importing this module puts code/ on sys.path (the modules there are flat, not a package).
OfflineTestCase makes any network access fail the test. requires_dataset skips a test when
dataset/ is absent, as it is in the submitted code.zip.
"""
import csv
import socket
import sys
import unittest
from pathlib import Path
from unittest import mock

CODE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

requires_dataset = unittest.skipUnless(
    (DATASET_DIR / "evidence_files.csv").exists(), "dataset/ is not present (it is not part of code.zip)"
)


class OfflineTestCase(unittest.TestCase):
    """A test case in which opening a socket or resolving a name is an error."""

    def setUp(self) -> None:
        super().setUp()
        for target in ("socket.socket.connect", "socket.getaddrinfo"):
            patcher = mock.patch(target, side_effect=AssertionError("network access attempted in a unit test"))
            patcher.start()
            self.addCleanup(patcher.stop)


def forbid_api_calls(test: unittest.TestCase) -> None:
    """Make any model call that is not served from the cache fail the test with AssertionError.

    The socket guard alone is not enough: the Anthropic client turns a connection error into
    LLMUnavailable, which callers treat as "use the fallback", so a cache miss would pass quietly.
    """
    import llm

    patcher = mock.patch.object(llm, "_call", side_effect=AssertionError("model call that missed the cache"))
    patcher.start()
    test.addCleanup(patcher.stop)


def read_case(requests_csv: Path, case_id: str) -> dict:
    with open(str(requests_csv), newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] == case_id:
                return row
    raise KeyError("%s is not in %s" % (case_id, requests_csv))
