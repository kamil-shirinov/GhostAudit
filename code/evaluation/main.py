#!/usr/bin/env python3
"""Score the agent on the public sample requests.

Runs predict() over dataset/sample_deletion_requests.csv, writes the predictions to
code/evaluation/sample_predictions.csv and scores them against the gold answers in
dataset/sample_output.csv. This is the only script that reads gold answers; the agent
never does. A summary is printed and written to code/evaluation/latest_metrics.md.

Usage: python3 code/evaluation/main.py [--case CASE_ID] [--intent]
--case also prints that case's missing and extra steps.
--intent also reports the intent classifier: probe accuracy for the LLM and the keyword
fallback, intent on the samples, and the NOT_APPLICABLE count among the real requests.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import agent  # noqa: E402
import intent  # noqa: E402
import llm  # noqa: E402
import main as pipeline  # noqa: E402  (code/main.py: this file runs as __main__, so no clash)
from intent_cases import PROBES  # noqa: E402
from validate import STATUSES  # noqa: E402

SAMPLE_REQUESTS = REPO_ROOT / "dataset" / "sample_deletion_requests.csv"
EVALUATION_REQUESTS = REPO_ROOT / "dataset" / "deletion_requests.csv"
GOLD = REPO_ROOT / "dataset" / "sample_output.csv"
PREDICTIONS = EVAL_DIR / "sample_predictions.csv"
METRICS_MD = EVAL_DIR / "latest_metrics.md"

# step_no is ignored when comparing steps: it is fully determined by position.
FIELDS = ("phase", "operation", "state", "system_id", "resource_id", "evidence_id")

Signature = Tuple[str, ...]
Cases = Dict[str, Tuple[str, List[dict]]]


def load_cases(path) -> Cases:
    """case_id -> (deletion_status, steps) from an output-format CSV."""
    cases = {}
    with open(str(path), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cases[row["case_id"]] = (row["deletion_status"], json.loads(row["steps_json"]))
    return cases


def signature(step: dict) -> Signature:
    return tuple(step.get(field) for field in FIELDS)


def multiset_diff(left: List[Signature], right: List[Signature]) -> List[Signature]:
    """Items of left with no unused equal item in right, in left order."""
    budget = Counter(right)
    leftover = []
    for item in left:
        if budget[item] > 0:
            budget[item] -= 1
        else:
            leftover.append(item)
    return leftover


def fraction(numerator: int, denominator: int) -> str:
    """'17/32 (53.1%)'. An empty denominator is reported as 0.0%."""
    percent = 100.0 * numerator / denominator if denominator else 0.0
    return "%d/%d (%.1f%%)" % (numerator, denominator, percent)


def compute_metrics(predictions: Cases, gold: Cases) -> List[Tuple[str, str]]:
    """(label, value) rows. The gold file defines the case set; a missing prediction is an error."""
    status_total = Counter()
    status_correct = Counter()
    matched = predicted = expected = 0
    position_total = 0
    field_correct = Counter()
    sequence_correct = 0

    for case_id, (gold_status, gold_steps) in gold.items():
        predicted_status, predicted_steps = predictions[case_id]
        status_total[gold_status] += 1
        if predicted_status == gold_status:
            status_correct[gold_status] += 1

        gold_sigs = [signature(step) for step in gold_steps]
        predicted_sigs = [signature(step) for step in predicted_steps]
        matched += len(gold_sigs) - len(multiset_diff(gold_sigs, predicted_sigs))
        predicted += len(predicted_sigs)
        expected += len(gold_sigs)

        for position in range(max(len(gold_sigs), len(predicted_sigs))):
            position_total += 1
            if position < len(gold_sigs) and position < len(predicted_sigs):
                for number, field in enumerate(FIELDS):
                    if gold_sigs[position][number] == predicted_sigs[position][number]:
                        field_correct[field] += 1

        if gold_sigs == predicted_sigs:
            sequence_correct += 1

    cases = len(gold)
    f1 = 2.0 * matched / (predicted + expected) if predicted + expected else 0.0
    rows = [
        ("Cases", str(cases)),
        ("Gold steps / predicted steps", "%d / %d" % (expected, predicted)),
        ("Status accuracy", fraction(sum(status_correct.values()), cases)),
    ]
    for status in STATUSES:
        rows.append(("Status accuracy, gold %s" % status, fraction(status_correct[status], status_total[status])))
    rows += [
        ("Step precision", fraction(matched, predicted)),
        ("Step recall", fraction(matched, expected)),
        ("Step F1", "%.1f%%" % (100.0 * f1)),
    ]
    for field in FIELDS:
        rows.append(("Field accuracy, %s" % field, fraction(field_correct[field], position_total)))
    rows.append(("Sequence accuracy", fraction(sequence_correct, cases)))
    return rows


def workflow_link_counts(gold: Cases) -> Tuple[int, int, int, int]:
    """(needed, correct, wrong, unmatched) for the workflow-linked accounts of the sample requests."""
    assignment = agent.workflow_assignment()
    needed = [case_id for case_id in gold if case_id in assignment.needed]
    correct = wrong = unmatched = 0
    for case_id in needed:
        link = assignment.links.get(case_id)
        if link is None:
            unmatched += 1
        elif link.resource_id in {step["resource_id"] for step in gold[case_id][1]}:
            correct += 1
        else:
            wrong += 1
    return len(needed), correct, wrong, unmatched


def workflow_link_row(gold: Cases) -> Tuple[str, str]:
    """How the workflow-linked accounts of the sample requests compare with gold's resources."""
    needed, correct, wrong, unmatched = workflow_link_counts(gold)
    return (
        "Workflow links",
        "%d needed an account: %d assigned correctly, %d assigned wrongly, %d unmatched"
        % (needed, correct, wrong, unmatched),
    )


def render_text(rows: List[Tuple[str, str]]) -> str:
    width = max(len(label) for label, _ in rows)
    return "\n".join("%s  %s" % (label.ljust(width), value) for label, value in rows)


def render_markdown(rows: List[Tuple[str, str]]) -> str:
    lines = [
        "# Latest evaluation metrics",
        "",
        "`code/agent.py` run on `dataset/sample_deletion_requests.csv` and scored against `dataset/sample_output.csv`.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    lines.extend("| %s | %s |" % (label, value) for label, value in rows)
    lines += [
        "",
        "- Steps are matched as a multiset of (phase, operation, state, system_id, resource_id, evidence_id);"
        " step_no is ignored. Precision, recall and F1 are micro-averaged over all cases.",
        "- Field accuracy compares steps position by position up to the longer list; a position present in only"
        " one list counts as wrong.",
        "- Sequence accuracy is the share of cases whose whole steps array equals gold exactly and in order.",
        "- Workflow links: sample deletions with no identifier-linked account; an assigned account is correct"
        " if gold's steps name that resource.",
        "- A metric with an empty denominator is reported as 0.0%.",
        "",
    ]
    return "\n".join(lines)


def describe(sig: Signature) -> str:
    phase, operation, state, system_id, resource_id, evidence_id = sig
    return "%s %s %s  %s/%s  evidence=%s" % (phase, operation, state, system_id, resource_id, evidence_id or "(none)")


def print_case(case_id: str, predictions: Cases, gold: Cases) -> None:
    gold_status, gold_steps = gold[case_id]
    predicted_status, predicted_steps = predictions[case_id]
    verdict = "match" if gold_status == predicted_status else "MISMATCH"
    print("\n%s: status gold=%s predicted=%s (%s)" % (case_id, gold_status, predicted_status, verdict))
    gold_sigs = [signature(step) for step in gold_steps]
    predicted_sigs = [signature(step) for step in predicted_steps]
    for title, sigs in (
        ("Missing (in gold, not predicted)", multiset_diff(gold_sigs, predicted_sigs)),
        ("Extra (predicted, not in gold)", multiset_diff(predicted_sigs, gold_sigs)),
    ):
        print("%s: %d" % (title, len(sigs)))
        for sig in sigs:
            print("  " + describe(sig))


def print_intent_report() -> None:
    """Probe accuracy for the LLM and the keyword fallback, then intent on the real requests."""
    print("\nIntent probes (D = deletion request, N = not)")
    llm_verdicts = intent.classify_intents([probe.text for probe in PROBES])
    llm_correct = keyword_correct = llm_answers = 0
    for number, probe in enumerate(PROBES, 1):
        llm = llm_verdicts[probe.text]
        keyword = intent.keyword_intent(probe.text)
        llm_answers += llm["source"] == "llm"
        llm_correct += llm["deletion"] == probe.deletion
        keyword_correct += keyword["deletion"] == probe.deletion
        print(
            "  %2d [%s] expected=%s  llm=%s%s  keyword=%s%s  %s"
            % (
                number,
                probe.language,
                "D" if probe.deletion else "N",
                "D" if llm["deletion"] else "N",
                "" if llm["deletion"] == probe.deletion else " WRONG",
                "D" if keyword["deletion"] else "N",
                "" if keyword["deletion"] == probe.deletion else " WRONG",
                probe.note,
            )
        )
    print("  LLM probe accuracy:      %s (%d answered by the model)" % (fraction(llm_correct, len(PROBES)), llm_answers))
    print("  Keyword probe accuracy:  %s" % fraction(keyword_correct, len(PROBES)))

    samples = pipeline.read_requests(SAMPLE_REQUESTS)
    agent.prepare(samples)
    verdicts = [agent.intent_for(case["request_text"]) for case in samples]
    print(
        "\nIntent on the %d sample requests (all deletions): %s classified as deletion (%d from the model)"
        % (len(samples), fraction(sum(v["deletion"] for v in verdicts), len(samples)), sum(v["source"] == "llm" for v in verdicts))
    )

    requests = pipeline.read_requests(EVALUATION_REQUESTS)
    agent.prepare(requests)
    refused = [(case, agent.intent_for(case["request_text"])) for case in requests]
    refused = [(case, verdict) for case, verdict in refused if not verdict["deletion"]]
    print("\nNOT_APPLICABLE predictions among the %d evaluation requests: %d" % (len(requests), len(refused)))
    for case, verdict in refused:
        print("  %s (%s): %s | %s" % (case["case_id"], verdict["source"], verdict["reason"], case["request_text"][:110]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score the agent on the sample requests.")
    parser.add_argument("--case", metavar="CASE_ID", help="also print this sample case's missing and extra steps")
    parser.add_argument(
        "--intent", action="store_true", help="also report intent probe accuracy (LLM vs keyword) and intent on the real requests"
    )
    args = parser.parse_args(argv)
    llm.start_run("evaluation")

    gold = load_cases(GOLD)
    if args.case is not None and args.case not in gold:
        parser.error("%s is not a case_id in %s" % (args.case, GOLD.name))

    try:
        rows = pipeline.run_agent(pipeline.read_requests(SAMPLE_REQUESTS))
    except (pipeline.KeywordFallbackError, pipeline.MissingVisionError) as exc:
        sys.stderr.write("ERROR: %s\nFix the cause (API key, spend cap, network) and re-run.\n" % exc)
        return 1
    pipeline.write_output(rows, PREDICTIONS)
    predictions = load_cases(PREDICTIONS)

    rows = compute_metrics(predictions, gold) + [workflow_link_row(gold)]
    print(render_text(rows))
    with open(str(METRICS_MD), "w", newline="\n", encoding="utf-8") as handle:
        handle.write(render_markdown(rows))
    print("\nWrote %s and %s" % (PREDICTIONS, METRICS_MD))

    if args.intent:
        print_intent_report()
    if args.case is not None:
        print_case(args.case, predictions, gold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
