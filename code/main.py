#!/usr/bin/env python3
"""Run the agent over a requests CSV and write output.csv (case_id,deletion_status,steps_json).

Usage: python3 code/main.py [--requests PATH] [--out PATH] [--allow-keyword-fallback]
                             [--allow-missing-vision]
       Exits with an error, writing nothing, if any request's intent came from the keyword
       fallback (LLM unavailable) unless --allow-keyword-fallback is given, or if any
       screenshot could not be read (no cached reading, LLM unavailable) unless
       --allow-missing-vision is given.
       python3 code/main.py --explain CASE_ID   (prints one case's audit trail; writes nothing)
Defaults come from the repository root, so this works from any working directory.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import evidence
import llm
from agent import audit_case, format_trail, intent_for, predict, prepare
from validate import HEADER, case_sort_key

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REQUESTS = REPO_ROOT / "dataset" / "deletion_requests.csv"
SAMPLE_REQUESTS = REPO_ROOT / "dataset" / "sample_deletion_requests.csv"
DEFAULT_OUT = REPO_ROOT / "output.csv"

Row = Tuple[str, str, List[dict]]

MAX_CASE_IDS_SHOWN = 30


class KeywordFallbackError(RuntimeError):
    """Some request intents came from the keyword fallback, not the language model."""

    def __init__(self, case_ids: List[str]) -> None:
        shown = ", ".join(case_ids[:MAX_CASE_IDS_SHOWN])
        more = len(case_ids) - MAX_CASE_IDS_SHOWN
        super().__init__(
            "%d request(s) were classified by the keyword fallback because the language model was unavailable "
            "(the reason is in the intent warning above): %s%s. No output was written."
            % (len(case_ids), shown, " ... and %d more" % more if more > 0 else "")
        )
        self.case_ids = case_ids


class MissingVisionError(RuntimeError):
    """Some screenshots could not be read, so their evidence would silently be missing."""

    def __init__(self, unreadable: List[Tuple[str, str]]) -> None:
        reasons = sorted({reason for _, reason in unreadable})
        super().__init__(
            "%d screenshot(s) could not be read (%s): %s. No output was written."
            % (len(unreadable), "; ".join(reasons), ", ".join(file_id for file_id, _ in unreadable))
        )
        self.unreadable = unreadable


def read_requests(path) -> List[Dict[str, str]]:
    with open(str(path), newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def renumber(steps: List[dict]) -> List[dict]:
    """Number steps 1..n by position; step_no from the agent is never trusted."""
    renumbered = []
    for number, step in enumerate(steps, 1):
        numbered = {"step_no": number}
        numbered.update((key, value) for key, value in step.items() if key != "step_no")
        renumbered.append(numbered)
    return renumbered


def run_agent(
    requests: List[Dict[str, str]], allow_keyword_fallback: bool = False, allow_missing_vision: bool = False
) -> List[Row]:
    """Call predict() once per request; rows come back sorted by the number in case_id.

    Raises KeywordFallbackError if any intent came from the keyword fallback, unless allowed:
    a silent fallback would turn, say, a German deletion into NOT_APPLICABLE. Raises
    MissingVisionError if any screenshot could not be read, unless allowed: its findings
    would silently disappear.
    """
    prepare(requests)  # classify all request intents once, each in its own cached call
    if not allow_keyword_fallback:
        fallback = [case["case_id"] for case in requests if intent_for(case["request_text"])["source"] == "keyword"]
        if fallback:
            raise KeywordFallbackError(fallback)
    unreadable = evidence.unreadable_screenshots()
    if unreadable and not allow_missing_vision:
        raise MissingVisionError(unreadable)
    rows = []
    for case in requests:
        status, steps = predict(case)
        rows.append((case["case_id"], status, renumber(steps)))
    rows.sort(key=lambda row: case_sort_key(row[0]))
    return rows


def serialise_steps(steps: List[dict]) -> str:
    return json.dumps(steps, ensure_ascii=False, separators=(",", ":"))


def write_output(rows: List[Row], path) -> None:
    with open(str(path), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        for case_id, status, steps in rows:
            writer.writerow([case_id, status, serialise_steps(steps)])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the deletion-audit agent and write output.csv.")
    parser.add_argument(
        "--requests",
        default=str(DEFAULT_REQUESTS),
        help="requests CSV to answer (default: dataset/deletion_requests.csv)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output CSV to write (default: output.csv)")
    parser.add_argument(
        "--allow-keyword-fallback",
        action="store_true",
        help="accept intent verdicts from the keyword fallback when the LLM is unavailable (default: fail)",
    )
    parser.add_argument(
        "--allow-missing-vision",
        action="store_true",
        help="run without screenshots that could not be read when the LLM is unavailable (default: fail)",
    )
    parser.add_argument(
        "--explain",
        metavar="CASE_ID",
        help="print the audit trail of one case from --requests or the sample requests, and exit",
    )
    args = parser.parse_args(argv)
    # Tags every API call of this process in the usage ledger. Only a full run is "main": it is the
    # one the usage report describes, and --explain must not displace it.
    llm.start_run("explain" if args.explain else "main")

    if args.explain:
        for path in (args.requests, SAMPLE_REQUESTS):
            cases = read_requests(path)
            for case in cases:
                if case["case_id"] == args.explain:
                    prepare(cases)  # same batches as a full run of this file, so the LLM cache is hit
                    print(format_trail(audit_case(case)))
                    for file_id, reason in evidence.unreadable_screenshots():
                        sys.stderr.write("WARNING: screenshot %s was not read (%s)\n" % (file_id, reason))
                    return 0
        parser.error("%s is in neither %s nor %s" % (args.explain, args.requests, SAMPLE_REQUESTS))

    try:
        rows = run_agent(read_requests(args.requests), args.allow_keyword_fallback, args.allow_missing_vision)
    except KeywordFallbackError as exc:
        sys.stderr.write(
            "ERROR: %s\nFix the cause (API key, spend cap, network) and re-run, "
            "or pass --allow-keyword-fallback to accept keyword verdicts.\n" % exc
        )
        return 1
    except MissingVisionError as exc:
        sys.stderr.write(
            "ERROR: %s\nFix the cause (API key, spend cap, network) and re-run, "
            "or pass --allow-missing-vision to run without them.\n" % exc
        )
        return 1
    write_output(rows, args.out)
    llm.record_run(requests=len(rows))

    counts = Counter(status for _, status, _ in rows)
    breakdown = ", ".join("%s=%d" % (status, counts[status]) for status in sorted(counts))
    print("Wrote %d rows to %s (%s)" % (len(rows), args.out, breakdown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
