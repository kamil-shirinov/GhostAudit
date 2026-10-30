"""Fail-loud guards: no model answer means an error, not a silent downgrade.

Every test runs against a scratch copy of the LLM cache, ledger and run log with the spend
cap at zero, so nothing under code/cache/ is touched and no API call can be made.
"""
import contextlib
import csv
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import common
import agent
import intent
import llm
import main

# A request text that is certainly not in the committed cache, and has no deletion verb.
UNCACHED_TEXT = "Could you send me a copy of the data you hold about me? (unit-test text, never cached)"
REQUEST_HEADER = ["case_id", "email", "requested_at", "reported_status", "reported_at", "audit_at", "request_text"]


@common.requires_dataset
class FailLoudTests(common.OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        # The real cache is copied so the screenshots stay readable; only UNCACHED_TEXT misses.
        shutil.copy(str(llm.CACHE_PATH), str(self.dir / "llm_cache.json"))

        patches = [
            mock.patch.dict(os.environ, {"LLM_SPEND_CAP_USD": "0"}),
            mock.patch.object(llm, "CACHE_DIR", self.dir),
            mock.patch.object(llm, "CACHE_PATH", self.dir / "llm_cache.json"),
            mock.patch.object(llm, "LEDGER_PATH", self.dir / "llm_usage.jsonl"),
            mock.patch.object(llm, "RUNS_PATH", self.dir / "llm_runs.jsonl"),
            mock.patch.object(llm, "_cache", None),  # reload from the scratch copy
            mock.patch.object(llm, "_run", None),  # main() starts a run; do not leak it
            mock.patch.object(llm, "_get_client", return_value=object()),  # reach the cap check without a key
            mock.patch.object(agent, "_intents", {}),  # fresh verdict memo
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_requests(self) -> Path:
        path = self.dir / "requests.csv"
        with open(str(path), "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(REQUEST_HEADER)
            writer.writerow(
                ["case_900", "nobody.at.all@example.org", "2026-09-01T00:00:00Z", "NOT_STARTED",
                 "2026-09-02T00:00:00Z", "2026-09-03T00:00:00Z", UNCACHED_TEXT]
            )
        return path

    def run_main(self, *flags):
        out = self.dir / "output.csv"
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main.main(["--requests", str(self.write_requests()), "--out", str(out)] + list(flags))
        return code, out, stderr.getvalue()

    def test_intent_raises_when_the_spend_cap_blocks_an_uncached_text(self) -> None:
        with self.assertRaises(llm.LLMUnavailable) as caught:
            intent._classify_batch([UNCACHED_TEXT])
        self.assertIn("spend cap", str(caught.exception))
        self.assertFalse((self.dir / "llm_usage.jsonl").exists(), "no call should have been recorded")

    def test_classify_intents_reports_the_keyword_fallback_it_used(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            verdicts = intent.classify_intents([UNCACHED_TEXT])
        self.assertEqual(verdicts[UNCACHED_TEXT]["source"], "keyword")
        self.assertIn("keyword fallback", stderr.getvalue())

    def test_main_refuses_to_write_output_on_the_keyword_fallback(self) -> None:
        code, out, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertFalse(out.exists(), "output.csv must not be written")
        self.assertIn("keyword fallback", stderr)
        self.assertIn("case_900", stderr)

    def test_main_writes_output_only_when_the_fallback_is_allowed(self) -> None:
        code, out, _ = self.run_main("--allow-keyword-fallback")
        self.assertEqual(code, 0)
        with open(str(out), newline="", encoding="utf-8") as handle:
            self.assertEqual(list(csv.reader(handle)), [["case_id", "deletion_status", "steps_json"], ["case_900", "NOT_APPLICABLE", "[]"]])


if __name__ == "__main__":
    unittest.main()
