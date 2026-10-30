#!/usr/bin/env python3
"""Validate output.csv against the Ghost Data Still Exists output contract.

validate_output() returns (errors, warnings). Errors are contract violations that the
grader may reject or mis-score. Warnings are legal but suspicious step lists, mostly
about citing evidence. Both are lists of (kind, case_id, detail); case_id is None for
file-level findings. Each kind is reported at most once per case, so a count is a number
of cases.

CLI: python3 code/validate.py [output.csv] [requests.csv]
Exits 1 if there is any error; warnings never change the exit code.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "output.csv"
DEFAULT_REQUESTS = REPO_ROOT / "dataset" / "deletion_requests.csv"

HEADER = ["case_id", "deletion_status", "steps_json"]
STATUSES = ("COMPLETE", "INCOMPLETE", "NOT_APPLICABLE")
PHASES = ("FINDING", "REMEDIATION", "VERIFICATION")
STEP_KEYS = ("step_no", "phase", "system_id", "resource_id", "operation", "state", "evidence_id")
STRING_KEYS = tuple(key for key in STEP_KEYS if key != "step_no")

# operation -> (phase, state), in the order of "Allowed Operations" in problem_statement.md.
ALLOWED_OPERATIONS = {
    "REPORT_REMAINING_RECORD": ("FINDING", "PRESENT"),
    "REPORT_EXPOSED_PII": ("FINDING", "PRESENT"),
    "REPORT_ACTIVE_DATA_FLOW": ("FINDING", "ACTIVE"),
    "REPORT_RESTORE_RISK": ("FINDING", "ACTIVE"),
    "DISABLE_DATA_FLOW": ("REMEDIATION", "DISABLED"),
    "BLOCK_RESTORE": ("REMEDIATION", "BLOCKED"),
    "DELETE_RECORD": ("REMEDIATION", "ABSENT"),
    "REDACT_RECORD": ("REMEDIATION", "REDACTED"),
    "ANONYMIZE_RECORD": ("REMEDIATION", "ANONYMIZED"),
    "VERIFY_RECORD_ABSENT": ("VERIFICATION", "ABSENT"),
    "VERIFY_RECORD_PRESENT": ("VERIFICATION", "PRESENT"),
    "VERIFY_RECORD_REDACTED": ("VERIFICATION", "REDACTED"),
    "VERIFY_RECORD_ANONYMIZED": ("VERIFICATION", "ANONYMIZED"),
    "VERIFY_DATA_FLOW_DISABLED": ("VERIFICATION", "DISABLED"),
    "VERIFY_RESTORE_BLOCKED": ("VERIFICATION", "BLOCKED"),
}

OPERATION_ORDER = {operation: number for number, operation in enumerate(ALLOWED_OPERATIONS)}

Finding = Tuple[str, Optional[str], str]

_CASE_ID = re.compile(r"case_(\d+)")


def case_sort_key(case_id: str) -> int:
    """The number after "case_". A plain string sort would put case_100 before case_33."""
    match = _CASE_ID.fullmatch(case_id)
    if match is None:
        raise ValueError("case_id %r is not of the form case_<number>" % case_id)
    return int(match.group(1))


def _check_case(status: str, raw_steps: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return ({error kind: detail}, {warning kind: detail}) for one output row."""
    errors = {}  # type: Dict[str, str]
    warnings = {}  # type: Dict[str, str]

    if status not in STATUSES:
        errors.setdefault("invalid_status", "deletion_status is %r" % status)

    try:
        steps = json.loads(raw_steps)
    except ValueError as exc:
        errors.setdefault("invalid_json", str(exc))
        return errors, warnings
    if not isinstance(steps, list):
        errors.setdefault("steps_not_array", "steps_json parses to %s, not a list" % type(steps).__name__)
        return errors, warnings

    if status == "NOT_APPLICABLE" and steps:
        errors.setdefault("not_applicable_with_steps", "NOT_APPLICABLE has %d steps" % len(steps))

    phase_indexes = []
    sort_keys = []  # (phase, operation order, system_id, resource_id, position) of well-formed steps
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            errors.setdefault("invalid_step", "step %d is not a JSON object" % index)
            continue
        keys = set(step)
        if keys != set(STEP_KEYS):
            errors.setdefault(
                "bad_step_keys",
                "step %d: missing %s, unexpected %s"
                % (index, sorted(set(STEP_KEYS) - keys), sorted(keys - set(STEP_KEYS))),
            )
            continue

        if type(step["step_no"]) is not int or step["step_no"] != index:
            errors.setdefault("bad_step_no", "step at position %d has step_no %r" % (index, step["step_no"]))

        not_strings = [key for key in STRING_KEYS if not isinstance(step[key], str)]
        if not_strings:
            errors.setdefault("bad_field_type", "step %d: %s must be a string" % (index, ", ".join(not_strings)))
            continue

        operation, phase, state = step["operation"], step["phase"], step["state"]
        if phase in PHASES:
            phase_indexes.append(PHASES.index(phase))
            if operation in ALLOWED_OPERATIONS:
                sort_keys.append(
                    (phase, OPERATION_ORDER[operation], step["system_id"], step["resource_id"], index)
                )
        if operation in ALLOWED_OPERATIONS:
            expected_phase, expected_state = ALLOWED_OPERATIONS[operation]
            if phase != expected_phase:
                errors.setdefault(
                    "phase_mismatch", "step %d: %s belongs to %s, not %s" % (index, operation, expected_phase, phase)
                )
            if state != expected_state:
                errors.setdefault(
                    "state_mismatch", "step %d: %s has state %s, not %s" % (index, operation, expected_state, state)
                )
        else:
            errors.setdefault("unknown_operation", "step %d: operation %r is not allowed" % (index, operation))
            if phase not in PHASES:
                errors.setdefault("unknown_phase", "step %d: phase %r is not allowed" % (index, phase))

    if phase_indexes != sorted(phase_indexes):
        errors.setdefault("phase_order", "phases are not in FINDING, REMEDIATION, VERIFICATION order")
    # Within a phase: Allowed Operations table order, then system_id, then resource_id.
    for previous, current in zip(sort_keys, sort_keys[1:]):
        if current[0] == previous[0] and current[1:4] < previous[1:4]:
            errors.setdefault(
                "within_phase_order", "step %d (%s/%s) should come before step %d" % (current[4], current[2], current[3], previous[4])
            )
            break

    dict_steps = [step for step in steps if isinstance(step, dict)]
    if status == "COMPLETE":
        if not steps:
            warnings["complete_no_steps"] = "COMPLETE with an empty steps array"
        if any(step.get("phase") != "VERIFICATION" for step in dict_steps):
            warnings["complete_non_verification_steps"] = "COMPLETE with a FINDING or REMEDIATION step"
        if any(not step.get("evidence_id") for step in dict_steps):
            warnings["complete_empty_evidence_id"] = "COMPLETE with a step that has an empty evidence_id"
    elif status == "INCOMPLETE":
        if any(step.get("phase") == "FINDING" and not step.get("evidence_id") for step in dict_steps):
            warnings["incomplete_finding_empty_evidence_id"] = "INCOMPLETE finding with an empty evidence_id"
        if any(step.get("phase") in ("REMEDIATION", "VERIFICATION") and step.get("evidence_id") for step in dict_steps):
            warnings["incomplete_future_step_has_evidence_id"] = (
                "INCOMPLETE remediation or verification step with a non-empty evidence_id"
            )
    return errors, warnings


def validate_output(output_path, requests_path) -> Tuple[List[Finding], List[Finding]]:
    """Check output_path against the contract and the case_ids in requests_path."""
    try:
        with open(str(output_path), newline="", encoding="utf-8") as handle:
            table = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return [("unreadable_output", None, "%s: %s" % (output_path, exc))], []

    if not table or table[0] != HEADER:
        got = ",".join(table[0]) if table else "an empty file"
        return [("wrong_header", None, "expected %s, got %s" % (",".join(HEADER), got))], []

    with open(str(requests_path), newline="", encoding="utf-8-sig") as handle:
        expected_ids = [row["case_id"] for row in csv.DictReader(handle)]

    file_errors = []  # type: List[Finding]
    row_errors = []  # type: List[Finding]
    warnings = []  # type: List[Finding]
    output_ids = []  # type: List[str]
    for row in table[1:]:
        if not row:
            continue  # blank line
        case_id = row[0]
        output_ids.append(case_id)
        if len(row) != len(HEADER):
            row_errors.append(("bad_column_count", case_id, "row has %d columns, expected %d" % (len(row), len(HEADER))))
            continue
        errors, case_warnings = _check_case(row[1], row[2])
        row_errors.extend((kind, case_id, detail) for kind, detail in errors.items())
        warnings.extend((kind, case_id, detail) for kind, detail in case_warnings.items())

    expected_set = set(expected_ids)
    output_set = set(output_ids)
    for case_id in expected_ids:
        if case_id not in output_set:
            file_errors.append(("missing_case_id", case_id, "no output row for this request"))
    reported = set()
    for case_id in output_ids:
        if case_id not in expected_set and case_id not in reported:
            file_errors.append(("extra_case_id", case_id, "not in the requests file"))
            reported.add(case_id)
    reported = set()
    seen = set()
    for case_id in output_ids:
        if case_id in seen and case_id not in reported:
            file_errors.append(("duplicate_case_id", case_id, "appears more than once"))
            reported.add(case_id)
        seen.add(case_id)

    keyed = []
    for case_id in output_ids:
        try:
            keyed.append((case_sort_key(case_id), case_id))
        except ValueError:
            pass  # a malformed id is already reported as extra
    for previous, current in zip(keyed, keyed[1:]):
        if current[0] < previous[0]:
            file_errors.append(("wrong_sort_order", current[1], "%s comes after %s" % (current[1], previous[1])))
            break

    return file_errors + row_errors, warnings


def _print_findings(title: str, findings: List[Finding], show_detail: bool) -> None:
    if not findings:
        print("%s: none" % title)
        return
    groups = {}  # type: Dict[str, List[Tuple[Optional[str], str]]]
    for kind, case_id, detail in findings:
        groups.setdefault(kind, []).append((case_id, detail))
    print("%s:" % title)
    for kind, items in groups.items():
        examples = items[:3]
        if show_detail:
            print("  %s: %d" % (kind, len(items)))
            for case_id, detail in examples:
                print("    %s%s" % (case_id + ": " if case_id else "", detail))
        else:
            print("  %s: %d (e.g. %s)" % (kind, len(items), ", ".join(str(case_id) for case_id, _ in examples)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate an output.csv against the output contract.")
    parser.add_argument("output", nargs="?", default=str(DEFAULT_OUTPUT), help="predictions CSV (default: output.csv)")
    parser.add_argument(
        "requests",
        nargs="?",
        default=str(DEFAULT_REQUESTS),
        help="requests CSV the output must cover (default: dataset/deletion_requests.csv)",
    )
    args = parser.parse_args(argv)

    errors, warnings = validate_output(args.output, args.requests)
    print("Validated %s against %s" % (args.output, args.requests))
    _print_findings("Errors", errors, show_detail=True)
    _print_findings("Warnings", warnings, show_detail=False)
    print("FAILED" if errors else "OK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
