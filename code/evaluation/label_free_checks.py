#!/usr/bin/env python3
"""Checks on the predictions for the evaluation requests that need no gold answers.

Each check compares the predictions with something else in dataset/ (the privacy intake's
CLASSIFY_REQUEST activities, the capture time of each cited file) or with the output
contract. They cannot say that an answer is right, only that it is not inconsistent:
- NOT_APPLICABLE verdicts that link, through identifiers, to an account the privacy intake
  also marked as non-deletion: a signal the intent classifier never sees;
- steps citing evidence captured after the case's audit_at;
- COMPLETE cases with a step that has an empty evidence_id;
- remediation steps with no matching verification step;
- the workflow's reported_status against our deletion_status.

Usage: python3 code/evaluation/label_free_checks.py [--output PATH]
Default: run the agent over dataset/deletion_requests.csv (model answers come from the cache).
--output checks an existing output.csv against the same requests instead.
Exits 1 if any check finds a violation.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import agent  # noqa: E402
import evidence  # noqa: E402
import llm  # noqa: E402
import main as pipeline  # noqa: E402  (code/main.py: CODE_DIR is first on the path)
from identity import resolve_identity  # noqa: E402
from validate import STATUSES  # noqa: E402
from wording import plural  # noqa: E402

EVALUATION_REQUESTS = REPO_ROOT / "dataset" / "deletion_requests.csv"
ACTIVITY_PATH = evidence.DATASET_DIR / "system_activity.csv"
AUDIT_LOG_PATH = evidence.DATASET_DIR / "user_audit_log.csv"

REPORTED_STATUSES = ("COMPLETE", "PARTIALLY_COMPLETE", "NOT_STARTED")

# remediation operation -> the verification step that must follow it, from the agent's own table.
VERIFIED_BY = {remediation: verification for _, remediation, verification in agent.OPERATIONS.values() if remediation}

Row = Tuple[str, str, List[dict]]  # case_id, deletion_status, steps
Violation = Tuple[str, str]  # case_id, what is wrong


class IntakeLink(NamedTuple):
    case_id: str
    status: str
    account: str
    activity_id: str
    activity_time: str
    in_window: bool  # the intake's mark falls between the request's requested_at and audit_at


class Checks(NamedTuple):
    cases: int
    matrix: Dict[Tuple[str, str], int]  # (reported_status, deletion_status) -> cases
    not_applicable: List[Tuple[str, List[str]]]  # (case_id, accounts it links to)
    agreeing: List[IntakeLink]  # NOT_APPLICABLE verdicts that link to an intake-marked account
    disagreeing: List[str]  # NOT_APPLICABLE case_ids that do not
    intake_accounts: Dict[str, str]  # account -> activity_id of the privacy intake's non-deletion mark
    intake_links: Dict[str, List[Tuple[str, str]]]  # account -> (case_id, deletion_status) of requests linked to it
    late_citations: List[Violation]
    unknown_citations: List[Violation]
    empty_complete: List[Violation]
    unmatched_remediation: List[Violation]

    def violations(self) -> int:
        return (
            len(self.disagreeing)
            + len(self.late_citations)
            + len(self.unknown_citations)
            + len(self.empty_complete)
            + len(self.unmatched_remediation)
        )


def _read_csv(path) -> List[Dict[str, str]]:
    with open(str(path), newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def evidence_times() -> Dict[str, str]:
    """evidence id -> when it exists: captured_at of a file, event_time of an activity or audit row."""
    times = {entry["file_id"]: entry["captured_at"] for entry in evidence.read_manifest()}
    for path, key in ((ACTIVITY_PATH, "activity_id"), (AUDIT_LOG_PATH, "audit_id")):
        times.update((row[key], row["event_time"]) for row in _read_csv(path))
    return times


def cited_late(audit_at: str, steps: List[dict], times: Dict[str, str]) -> Tuple[List[Tuple[int, str, str]], List[Tuple[int, str]]]:
    """(step_no, evidence_id, time) cited after audit_at, and (step_no, evidence_id) no source knows."""
    late, unknown = [], []
    for step in steps:
        cited = step["evidence_id"]
        if not cited:
            continue
        if cited not in times:
            unknown.append((step["step_no"], cited))
        elif times[cited] > audit_at:
            late.append((step["step_no"], cited, times[cited]))
    return late, unknown


def intake_marked_accounts() -> Dict[str, str]:
    """account -> activity_id, for each CLASSIFY_REQUEST the privacy intake closed as non-deletion."""
    marked = {}  # type: Dict[str, str]
    for row in _read_csv(ACTIVITY_PATH):
        if row["event_type"] == "CLASSIFY_REQUEST" and "non-deletion" in row["details"]:
            marked.setdefault(row["resource_id"], row["activity_id"])
    return marked


def read_rows(path) -> List[Row]:
    return [(row["case_id"], row["deletion_status"], json.loads(row["steps_json"])) for row in _read_csv(path)]


def run_checks(requests: List[Dict[str, str]], rows: List[Row]) -> Checks:
    by_id = {case["case_id"]: case for case in requests}
    if sorted(by_id) != sorted(case_id for case_id, _, _ in rows):
        raise ValueError("the predictions do not cover exactly the %s" % plural(len(by_id), "request"))

    index = agent._index()
    times = evidence_times()
    intake = intake_marked_accounts()
    intake_time = {row["activity_id"]: row["event_time"] for row in _read_csv(ACTIVITY_PATH)}

    matrix = Counter()  # type: Counter
    not_applicable, agreeing, disagreeing = [], [], []
    intake_links = {account: [] for account in intake}  # type: Dict[str, List[Tuple[str, str]]]
    late_citations, unknown_citations, empty_complete, unmatched = [], [], [], []

    for case_id, status, steps in rows:
        case = by_id[case_id]
        matrix[(case["reported_status"], status)] += 1

        # Accounts reached by identifiers alone: the workflow's activity stream is not used here.
        linked = agent._link_by_identity(resolve_identity(case["email"]), index)
        accounts = sorted(resource for resource in linked if evidence.system_for(resource) == "account_service")
        for account in accounts:
            if account in intake:
                intake_links[account].append((case_id, status))
        if status == "NOT_APPLICABLE":
            not_applicable.append((case_id, accounts))
            hits = [account for account in accounts if account in intake]
            if hits:
                activity_id = intake[hits[0]]
                marked_at = intake_time[activity_id]
                in_window = case["requested_at"] <= marked_at <= case["audit_at"]
                agreeing.append(IntakeLink(case_id, status, hits[0], activity_id, marked_at, in_window))
            else:
                disagreeing.append(case_id)

        late, unknown = cited_late(case["audit_at"], steps, times)
        late_citations.extend(
            (case_id, "step %d cites %s captured %s, after audit_at %s" % (no, cited, when, case["audit_at"]))
            for no, cited, when in late
        )
        unknown_citations.extend((case_id, "step %d cites %s, which is in no source" % item) for item in unknown)

        if status == "COMPLETE":
            empty_complete.extend(
                (case_id, "step %d %s on %s has an empty evidence_id" % (s["step_no"], s["operation"], s["resource_id"]))
                for s in steps
                if not s["evidence_id"]
            )

        verified = {(s["system_id"], s["resource_id"], s["operation"]) for s in steps if s["phase"] == "VERIFICATION"}
        for s in steps:
            if s["phase"] != "REMEDIATION":
                continue
            wanted = VERIFIED_BY.get(s["operation"])
            if wanted is None or (s["system_id"], s["resource_id"], wanted) not in verified:
                unmatched.append(
                    (case_id, "step %d %s on %s has no %s step" % (s["step_no"], s["operation"], s["resource_id"], wanted))
                )

    return Checks(
        len(rows), dict(matrix), not_applicable, agreeing, disagreeing, intake, intake_links,
        late_citations, unknown_citations, empty_complete, unmatched,
    )


def matrix_rows(checks: Checks) -> List[Tuple[str, List[int]]]:
    """(reported_status, counts per deletion_status in STATUSES order), canonical reported order first."""
    reported = {key[0] for key in checks.matrix}
    order = [status for status in REPORTED_STATUSES if status in reported] + sorted(reported - set(REPORTED_STATUSES))
    return [(status, [checks.matrix.get((status, audited), 0) for audited in STATUSES]) for status in order]


def _examples(violations: List[Violation], limit: int = 5) -> List[str]:
    lines = ["    %s: %s" % item for item in violations[:limit]]
    if len(violations) > limit:
        lines.append("    ... and %d more" % (len(violations) - limit))
    return lines


def render_text(checks: Checks) -> str:
    na = len(checks.not_applicable)
    lines = [
        "Label-free checks on %s" % plural(checks.cases, "prediction"),
        "",
        "NOT_APPLICABLE verdicts linked to an account the privacy intake also marked non-deletion: %d/%d" % (len(checks.agreeing), na),
    ]
    lines += [
        "    %s -> %s (%s, %s%s)" % (c.case_id, c.account, c.activity_id, c.activity_time, "" if c.in_window else ", OUTSIDE the request window")
        for c in checks.agreeing
    ]
    lines += ["    NO LINK: %s" % case_id for case_id in checks.disagreeing]
    marked = sum(1 for links in checks.intake_links.values() if any(status == "NOT_APPLICABLE" for _, status in links))
    lines.append(
        "Intake-marked accounts (%d) whose requester we answered NOT_APPLICABLE: %d" % (len(checks.intake_accounts), marked)
    )
    for title, found in (
        ("Steps citing evidence captured after audit_at (expect 0)", checks.late_citations),
        ("Steps citing an id found in no source (expect 0)", checks.unknown_citations),
        ("COMPLETE steps with an empty evidence_id (expect 0)", checks.empty_complete),
        ("Remediation steps without a matching verification step (expect 0)", checks.unmatched_remediation),
    ):
        lines.append("%s: %d" % (title, len(found)))
        lines += _examples(found)

    lines += ["", "Workflow reported_status (rows) against our deletion_status (columns)"]
    width = max(len(status) for status in REPORTED_STATUSES)
    lines.append("  %s  %s  total" % ("".ljust(width), "  ".join(status.rjust(14) for status in STATUSES)))
    for reported, counts in matrix_rows(checks):
        lines.append("  %s  %s  %5d" % (reported.ljust(width), "  ".join(str(n).rjust(14) for n in counts), sum(counts)))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check the predictions for the evaluation requests without labels.")
    parser.add_argument("--output", metavar="PATH", help="check this output.csv instead of running the agent")
    args = parser.parse_args(argv)
    llm.start_run("evaluation")

    requests = pipeline.read_requests(EVALUATION_REQUESTS)
    if args.output:
        rows = read_rows(args.output)
    else:
        try:
            rows = pipeline.run_agent(requests)
        except (pipeline.KeywordFallbackError, pipeline.MissingVisionError) as exc:
            sys.stderr.write("ERROR: %s\nFix the cause (API key, spend cap, network) and re-run.\n" % exc)
            return 1
    checks = run_checks(requests, rows)
    print(render_text(checks))
    return 1 if checks.violations() else 0


if __name__ == "__main__":
    sys.exit(main())
