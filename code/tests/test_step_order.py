"""Step ordering: phase, then operation-table order, then system_id, then resource_id."""
import csv
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import common  # noqa: F401  (puts code/ on sys.path)
from agent import step_sort_key
from validate import ALLOWED_OPERATIONS, HEADER, validate_output


def make_step(operation: str, system_id: str, resource_id: str) -> Dict[str, str]:
    phase, state = ALLOWED_OPERATIONS[operation]
    return {
        "step_no": 0,
        "phase": phase,
        "system_id": system_id,
        "resource_id": resource_id,
        "operation": operation,
        "state": state,
        "evidence_id": "",
    }


# Already in the required order. Each neighbouring pair is separated by exactly one rule:
# 0 -> 1 operation-table order beats system_id (search_service sorts after account_service),
# 1 -> 2 resource_id, 2 -> 3 system_id, 3 -> 4 phase beats system_id, 4 -> 5 operation-table order.
EXPECTED = [
    make_step("REPORT_REMAINING_RECORD", "search_service", "search_001"),
    make_step("REPORT_EXPOSED_PII", "account_service", "account_001"),
    make_step("REPORT_EXPOSED_PII", "account_service", "account_002"),
    make_step("REPORT_EXPOSED_PII", "analytics_platform", "analytics_001"),
    make_step("DELETE_RECORD", "account_service", "account_001"),
    make_step("VERIFY_RECORD_ABSENT", "account_service", "account_001"),
]


class StepSortKeyTests(unittest.TestCase):
    def test_every_permutation_sorts_into_the_required_order(self) -> None:
        for permutation in itertools.permutations(EXPECTED):
            self.assertEqual(sorted(permutation, key=step_sort_key), EXPECTED)

    def test_equal_keys_keep_their_original_order(self) -> None:
        # list.sort() is stable; the engine relies on it for two steps of the same resource and operation.
        first, second = dict(EXPECTED[1], state="PRESENT"), dict(EXPECTED[1], evidence_id="evidence_09")
        self.assertEqual(sorted([first, second], key=step_sort_key), [first, second])
        self.assertEqual(sorted([second, first], key=step_sort_key), [second, first])


class AgreesWithValidatorTests(unittest.TestCase):
    """The engine's order and the validator's order are written separately; they must not drift."""

    def validation_errors(self, steps: List[Dict[str, str]]):
        numbered = [dict(step, step_no=number) for number, step in enumerate(steps, 1)]
        with tempfile.TemporaryDirectory() as tmp:
            requests, output = Path(tmp) / "requests.csv", Path(tmp) / "output.csv"
            requests.write_text("case_id\ncase_1\n", encoding="utf-8")
            with open(str(output), "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(HEADER)
                writer.writerow(["case_1", "INCOMPLETE", json.dumps(numbered)])
            errors, _ = validate_output(output, requests)
        return errors

    def test_validator_accepts_the_engine_order(self) -> None:
        self.assertEqual(self.validation_errors(sorted(reversed(EXPECTED), key=step_sort_key)), [])

    def test_validator_rejects_the_reverse(self) -> None:
        self.assertNotEqual(self.validation_errors(list(reversed(EXPECTED))), [])


if __name__ == "__main__":
    unittest.main()
