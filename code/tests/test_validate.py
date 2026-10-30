"""validate.py rejects each kind of malformed output and accepts a minimal valid one."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Sequence

import common  # noqa: F401  (puts code/ on sys.path)
from validate import HEADER, validate_output

FINDING = ("FINDING", "REPORT_REMAINING_RECORD", "PRESENT")
REMEDIATION = ("REMEDIATION", "DELETE_RECORD", "ABSENT")
VERIFICATION = ("VERIFICATION", "VERIFY_RECORD_ABSENT", "ABSENT")


def step(step_no, kind, system_id="account_service", resource_id="account_001", evidence_id=None) -> Dict:
    phase, operation, state = kind
    if evidence_id is None:
        evidence_id = "evidence_01" if phase == "FINDING" else ""
    return {
        "step_no": step_no,
        "phase": phase,
        "system_id": system_id,
        "resource_id": resource_id,
        "operation": operation,
        "state": state,
        "evidence_id": evidence_id,
    }


def plan() -> List[Dict]:
    """A valid INCOMPLETE plan: one finding, its remediation, its verification."""
    return [step(1, FINDING), step(2, REMEDIATION), step(3, VERIFICATION)]


class ValidateOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def check(self, rows: Sequence[Sequence[str]], header: Sequence[str] = HEADER, expected_ids=("case_1", "case_2")):
        requests = self.dir / "requests.csv"
        with open(str(requests), "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["case_id", "email"])
            writer.writerows([case_id, "a@example.com"] for case_id in expected_ids)
        output = self.dir / "output.csv"
        with open(str(output), "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(list(header))
            writer.writerows(rows)
        return validate_output(output, requests)

    def error_kinds(self, rows, **kwargs):
        errors, _ = self.check(rows, **kwargs)
        return sorted({kind for kind, _, _ in errors})

    def test_minimal_valid_file_is_accepted(self) -> None:
        verified = [step(1, VERIFICATION, evidence_id="evidence_07")]
        errors, warnings = self.check(
            [
                ["case_1", "INCOMPLETE", json.dumps(plan())],
                ["case_2", "COMPLETE", json.dumps(verified)],
                ["case_3", "NOT_APPLICABLE", "[]"],
            ],
            expected_ids=("case_1", "case_2", "case_3"),
        )
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_wrong_header(self) -> None:
        kinds = self.error_kinds([["case_1", "NOT_APPLICABLE", "[]"]], header=["case_id", "status", "steps_json"])
        self.assertEqual(kinds, ["wrong_header"])

    def test_duplicate_case_id(self) -> None:
        kinds = self.error_kinds(
            [["case_1", "NOT_APPLICABLE", "[]"], ["case_1", "NOT_APPLICABLE", "[]"], ["case_2", "NOT_APPLICABLE", "[]"]]
        )
        self.assertEqual(kinds, ["duplicate_case_id"])

    def test_non_consecutive_step_no(self) -> None:
        steps = plan()
        steps[1]["step_no"] = 3
        steps[2]["step_no"] = 4
        kinds = self.error_kinds([["case_1", "INCOMPLETE", json.dumps(steps)], ["case_2", "NOT_APPLICABLE", "[]"]])
        self.assertEqual(kinds, ["bad_step_no"])

    def test_operation_in_the_wrong_phase(self) -> None:
        steps = [step(1, FINDING), step(2, REMEDIATION), step(3, VERIFICATION)]
        steps[1]["phase"] = "FINDING"  # DELETE_RECORD is a remediation
        kinds = self.error_kinds([["case_1", "INCOMPLETE", json.dumps(steps)], ["case_2", "NOT_APPLICABLE", "[]"]])
        self.assertIn("phase_mismatch", kinds)

    def test_phases_out_of_order(self) -> None:
        steps = [step(1, REMEDIATION), step(2, FINDING), step(3, VERIFICATION)]
        kinds = self.error_kinds([["case_1", "INCOMPLETE", json.dumps(steps)], ["case_2", "NOT_APPLICABLE", "[]"]])
        self.assertEqual(kinds, ["phase_order"])

    def test_wrong_order_within_a_phase(self) -> None:
        exposed = ("FINDING", "REPORT_EXPOSED_PII", "PRESENT")
        cases = {
            "operation-table order": [step(1, exposed), step(2, FINDING)],
            "system_id": [step(1, FINDING, "search_service", "search_001"), step(2, FINDING)],
            "resource_id": [step(1, FINDING, resource_id="account_002"), step(2, FINDING, resource_id="account_001")],
        }
        for name, steps in cases.items():
            with self.subTest(name):
                kinds = self.error_kinds(
                    [["case_1", "INCOMPLETE", json.dumps(steps)], ["case_2", "NOT_APPLICABLE", "[]"]]
                )
                self.assertEqual(kinds, ["within_phase_order"])

    def test_not_applicable_with_steps(self) -> None:
        kinds = self.error_kinds([["case_1", "NOT_APPLICABLE", json.dumps(plan())], ["case_2", "NOT_APPLICABLE", "[]"]])
        self.assertEqual(kinds, ["not_applicable_with_steps"])


if __name__ == "__main__":
    unittest.main()
