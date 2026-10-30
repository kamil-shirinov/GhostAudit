#!/usr/bin/env python3
"""Check evidence extraction (code/evidence.py).

Prints sighting counts per format, files with no sightings, the records read off each
screenshot, every alias-resolved sighting, unknown resource prefixes and
gold-citation support: a step in dataset/sample_output.csv with a non-empty evidence_id is
supported if that file has a sighting of the step's resource, or a coverage entry for the
step's system_id that does not list the resource.

This checks extraction only. It reads gold answers to measure support; the agent must
never read sample_output.csv.

Usage: python3 code/evaluation/check_evidence.py [--resource RESOURCE_ID]
Exits 1 if there are unreadable screenshots, unknown prefixes or unsupported gold citations.
"""
import argparse
import csv
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import evidence  # noqa: E402
import llm  # noqa: E402
import vision  # noqa: E402

GOLD = REPO_ROOT / "dataset" / "sample_output.csv"
STATE_FIELDS = ("status", "enabled", "restore_allowed", "present", "retention_end")


def print_screenshots(manifest, unreadable):
    """Every record each screenshot was read into (from the vision cache), then all resource IDs."""
    print("\nScreenshots")
    resource_ids = set()
    for entry in manifest:
        if Path(entry["file_path"]).suffix.lower() not in evidence.SCREENSHOT_SUFFIXES:
            continue
        if entry["file_id"] in unreadable:
            print("  %s  %s: UNREADABLE (%s)" % (entry["file_id"], entry["file_path"], unreadable[entry["file_id"]]))
            continue
        screenshot = vision.extract_screenshot(evidence.DATASET_DIR / entry["file_path"])
        print(
            "  %s  %s  screen=%r  records=%d"
            % (entry["file_id"], Path(entry["file_path"]).name, screenshot["screen"], len(screenshot["records"]))
        )
        for record in screenshot["records"]:
            fields = record["fields"]
            state = ", ".join("%s=%s" % (key, fields[key]) for key in STATE_FIELDS if key in fields)
            print(
                "    %-18s aliases=[%s]  %s"
                % (record["resource_id"] or "(document)", ", ".join(record["aliases"]), state)
            )
            if record["resource_id"]:
                resource_ids.add(record["resource_id"])
    print("  resource IDs: %s" % ", ".join(sorted(resource_ids, key=lambda rid: (evidence.resource_prefix(rid), rid))))


def print_alias_sightings(sightings, formats):
    resolved = [s for s in sightings if "alias" in s["context"]]
    print("\nAlias-resolved sightings: %d" % len(resolved))
    for s in resolved:
        print(
            "  %s (%s) -> %s via %r (%s)  observed_at=%s state=%s"
            % (
                s["evidence_id"],
                formats[s["evidence_id"]],
                s["resource_id"],
                s["context"]["alias"],
                s["context"]["alias_activity"],
                s["observed_at"],
                s["state"],
            )
        )


def print_resource(resource_id, sightings, formats):
    found = sorted((s for s in sightings if s["resource_id"] == resource_id), key=lambda s: s["observed_at"])
    print("Sightings of %s: %d" % (resource_id, len(found)))
    for s in found:
        print(
            "\n%s (%s) observed_at=%s captured_at=%s system=%s state=%s"
            % (s["evidence_id"], formats[s["evidence_id"]], s["observed_at"], s["captured_at"], s["system_id"], s["state"])
        )
        print("  context:     %s" % json.dumps(s["context"], ensure_ascii=False))
        print("  identifiers: %s" % ", ".join(s["identifiers"]))
        print("  linked:      %s" % ", ".join(s["linked_resource_ids"]))
        fields = json.dumps(s["fields"], ensure_ascii=False)
        print("  fields:      %s" % (fields if len(fields) <= 400 else fields[:400] + "..."))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check evidence extraction.")
    parser.add_argument("--resource", metavar="RESOURCE_ID", help="print every sighting of this resource")
    args = parser.parse_args(argv)
    llm.start_run("evaluation")

    manifest = evidence.read_manifest()
    formats = OrderedDict((entry["file_id"], Path(entry["file_path"]).suffix.lstrip(".").lower()) for entry in manifest)
    sightings = evidence.load_sightings()
    coverage = evidence.load_coverage()

    if args.resource:
        print_resource(args.resource, sightings, formats)
        return 0

    sightings_per_file = Counter(s["evidence_id"] for s in sightings)
    coverage_per_file = Counter(c["evidence_id"] for c in coverage)

    print("Sightings per format")
    print("  %-5s %6s %18s %10s %9s" % ("fmt", "files", "files w/ sightings", "sightings", "coverage"))
    for fmt in sorted(set(formats.values())):
        files = [file_id for file_id, f in formats.items() if f == fmt]
        print(
            "  %-5s %6d %18d %10d %9d"
            % (
                fmt,
                len(files),
                sum(1 for file_id in files if sightings_per_file[file_id]),
                sum(sightings_per_file[file_id] for file_id in files),
                sum(coverage_per_file[file_id] for file_id in files),
            )
        )
    print("  total: %d sightings, %d coverage entries" % (len(sightings), len(coverage)))
    unreadable = dict(evidence.unreadable_screenshots())
    print("  unreadable screenshots: %d" % len(unreadable))

    paths = {entry["file_id"]: entry["file_path"] for entry in manifest}
    empty = [file_id for file_id in formats if not sightings_per_file[file_id]]
    print("\nFiles with no sightings: %d" % len(empty))
    for file_id in empty:
        if file_id in unreadable:
            note = "unreadable screenshot"
        elif coverage_per_file[file_id]:
            note = "coverage only: %d entries" % coverage_per_file[file_id]
        else:
            note = "nothing extracted"
        print("  %s  %s  (%s)" % (file_id, paths[file_id], note))

    print_screenshots(manifest, unreadable)
    print_alias_sightings(sightings, formats)

    unknown = Counter()
    unknown_example = {}
    for s in sightings:
        if s["system_id"] is None:
            prefix = evidence.resource_prefix(s["resource_id"])
            unknown[prefix] += 1
            unknown_example.setdefault(prefix, "%s in %s" % (s["resource_id"], s["evidence_id"]))
    print("\nUnknown resource prefixes: %s" % ("none" if not unknown else len(unknown)))
    for prefix, count in unknown.most_common():
        print("  %s: %d (e.g. %s)" % (prefix, count, unknown_example[prefix]))

    seen = {(s["evidence_id"], s["resource_id"]) for s in sightings}
    total = 0
    by_sighting = by_coverage = 0
    unsupported = []
    with open(str(GOLD), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for step in json.loads(row["steps_json"]):
                evidence_id = step.get("evidence_id")
                if not evidence_id:
                    continue
                total += 1
                if (evidence_id, step["resource_id"]) in seen:
                    by_sighting += 1
                elif any(
                    c["evidence_id"] == evidence_id
                    and c["system_id"] == step["system_id"]
                    and step["resource_id"] not in c["resource_ids"]
                    for c in coverage
                ):
                    by_coverage += 1
                else:
                    reason = "not in the evidence manifest" if evidence_id not in formats else "no sighting or coverage"
                    unsupported.append((row["case_id"], step, reason))
    supported = by_sighting + by_coverage
    print(
        "\nGold-citation support: %d/%d (%d by sighting, %d by coverage)" % (supported, total, by_sighting, by_coverage)
    )
    for case_id, step, reason in unsupported:
        print(
            "  %s step %s: %s %s/%s cites %s (%s)"
            % (case_id, step["step_no"], step["operation"], step["system_id"], step["resource_id"], step["evidence_id"], reason)
        )

    return 1 if unreadable or unknown or unsupported else 0


if __name__ == "__main__":
    sys.exit(main())
