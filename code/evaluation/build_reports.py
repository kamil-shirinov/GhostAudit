#!/usr/bin/env python3
"""Write the two reports the submission needs: evaluation_report.md and usage_report.md.

Every number in them is computed here, in one run, from the evaluator, the custom cases,
the label-free checks and the usage ledger, and written straight into the Markdown, so the
reports cannot drift from the code. The agent's answers come from the response cache, so
the run makes no API call unless a prompt or an input changed.

Narrative sections are read from code/evaluation/narrative/{approach,failures,limitations}.md;
a missing file leaves a visible TODO in its place.

Usage: python3 code/evaluation/build_reports.py
"""
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import agent  # noqa: E402
import intent  # noqa: E402
import llm  # noqa: E402
import main as pipeline  # noqa: E402  (code/main.py: CODE_DIR is first on the path)
from validate import STATUSES  # noqa: E402

import custom_cases  # noqa: E402
import label_free_checks  # noqa: E402
from intent_cases import PROBES  # noqa: E402
from wording import plural  # noqa: E402

EVALUATION_REPORT = EVAL_DIR / "evaluation_report.md"
USAGE_REPORT = EVAL_DIR / "usage_report.md"
HISTORY_DIR = EVAL_DIR / "history"
NARRATIVE_DIR = EVAL_DIR / "narrative"

# Narrative file -> (heading used if the file is missing, what to write).
NARRATIVES = {
    "approach": ("Approach", "how the agent works and why it is built this way"),
    "failures": ("Failures found and improvements made", "each failure measured, the change made, and its effect"),
    "limitations": ("Known limitations", "what the agent cannot do or the evaluation cannot show"),
}

PROGRESSION_COLUMNS = ("Status accuracy", "Step precision", "Step recall", "Step F1", "Sequence accuracy")

Entry = Dict[str, Any]


# ---------------------------------------------------------------- Markdown helpers

def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], right: Sequence[int] = ()) -> str:
    """A GitHub table; the columns whose index is in right are right-aligned."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if number in right else "---" for number in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def percent_of(value: str) -> str:
    """'177/182 (97.3%)' -> '97.3%'; '94.4%' -> '94.4%'; an empty denominator -> 'n/a'."""
    if value.startswith("0/0"):
        return "n/a"
    found = re.search(r"(\d+(?:\.\d+)?%)", value)
    return found.group(1) if found else value


def shown(value: str) -> str:
    """The evaluator reports an empty denominator as 0.0%; the report says n/a."""
    return "n/a (no such cases)" if value.startswith("0/0") else value


def fraction(numerator: int, denominator: int) -> str:
    return "%d/%d (%.1f%%)" % (numerator, denominator, 100.0 * numerator / denominator if denominator else 0.0)


def narrative(name: str) -> str:
    path = NARRATIVE_DIR / (name + ".md")
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    heading, purpose = NARRATIVES[name]
    return "## %s\n\n> **TODO:** `code/evaluation/narrative/%s.md` is missing. Write it: %s." % (heading, name, purpose)


def load_sample_evaluator() -> Any:
    """code/evaluation/main.py under another name: 'main' is code/main.py here."""
    spec = importlib.util.spec_from_file_location("sample_evaluator", str(EVAL_DIR / "main.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- evaluation report

def history_metrics(path: Path) -> Dict[str, str]:
    """label -> value from a saved latest_metrics.md table."""
    metrics = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) == 2 and cells[0] not in ("Metric", "---"):
            metrics[cells[0]] = cells[1]
    return metrics


def history_order(path: Path) -> Tuple[int, str]:
    found = re.match(r"m(\d+)_", path.stem)
    return (int(found.group(1)) if found else 10 ** 6, path.stem)


def milestone_name(path: Path) -> str:
    name = re.sub(r"^m\d+_", "", path.stem).replace("_", " ")
    return name[:1].upper() + name[1:]


def progression_section(current: Dict[str, str]) -> str:
    rows = []  # type: List[List[str]]
    for path in sorted(HISTORY_DIR.glob("*.md"), key=history_order):
        metrics = history_metrics(path)
        rows.append([milestone_name(path)] + [percent_of(metrics.get(column, "n/a")) for column in PROGRESSION_COLUMNS])
    rows.append(["**Current**"] + [percent_of(current[column]) for column in PROGRESSION_COLUMNS])
    repeated = [row[0].strip("*") for previous, row in zip(rows, rows[1:]) if previous[1:] == row[1:]]
    lines = [
        "## Progression on the labelled samples",
        "",
        "Each milestone row is the evaluator's output saved when that milestone landed"
        " (`code/evaluation/history/`); the last row is this run.",
        "",
        table(("Milestone",) + PROGRESSION_COLUMNS, rows, right=range(1, len(PROGRESSION_COLUMNS) + 1)),
    ]
    if repeated:
        lines += [
            "",
            "A row equal to the one above it (%s) changed no answer on the labelled samples; the effect of those"
            " milestones is on the unlabelled evaluation requests (see `code/DECISIONS.md`)." % ", ".join(repeated),
        ]
    return "\n".join(lines)


def metrics_section(rows: List[Tuple[str, str]], links: Tuple[int, int, int, int], samples: int) -> str:
    needed, correct, wrong, unmatched = links
    body = [(label, shown(value)) for label, value in rows]
    body.append((
        "Workflow-link accuracy",
        "%s of the sample requests that needed an account assigned correctly (%d wrongly, %d unmatched)"
        % (fraction(correct, needed), wrong, unmatched),
    ))
    return "\n".join([
        "## Results on the %d labelled samples" % samples,
        "",
        table(("Metric", "Value"), body),
        "",
        "The samples contain no non-deletion request, so status accuracy for gold NOT_APPLICABLE is n/a here;"
        " that verdict is covered by the intent probes, the custom cases and the label-free checks below.",
    ])


def method_section(requests: int, samples: int, custom: int) -> str:
    return "\n".join([
        "## How we evaluate",
        "",
        "None of the %s has a known answer. The only labelled cases are the %s, so the evidence comes in three layers:"
        % (plural(requests, "evaluation request"), plural(samples, "separate sample request")),
        "",
        "1. **Development set: the %s.** `dataset/sample_deletion_requests.csv` is scored against"
        " `dataset/sample_output.csv` by `code/evaluation/main.py`, the only code that reads the gold answers; the agent"
        " never does. The samples were used to design and tune the rules, so they measure fit, not generalisation."
        % plural(samples, "labelled sample"),
        "2. **Custom cases.** %s (`code/evaluation/custom_cases.py`) change one thing in a sample"
        " request and check that only what should change does." % plural(custom, "metamorphic test"),
        "3. **Label-free checks on the %s** (`code/evaluation/label_free_checks.py`). They compare the"
        " predictions with the rest of `dataset/` and with the output contract; they cannot prove an answer right, only"
        " show it is not inconsistent." % plural(requests, "evaluation request"),
        "",
        "Metric definitions:",
        "",
        "- **Status accuracy**: the share of cases whose `deletion_status` equals gold, overall and per gold status.",
        "- **Step precision, recall, F1**: steps are compared as a multiset of (phase, operation, state, system_id,"
        " resource_id, evidence_id); `step_no` is ignored because position fixes it. Micro-averaged over all cases:"
        " precision = matched / predicted steps, recall = matched / gold steps, F1 = 2 x matched / (predicted + gold).",
        "- **Field accuracy**: steps are compared position by position up to the longer of the two lists; a field counts"
        " when gold and prediction agree at that position, and a position present in only one list counts as wrong.",
        "- **Sequence accuracy**: the share of cases whose whole `steps` array equals gold exactly and in order.",
        "- **Workflow-link accuracy**: of the sample requests whose identifiers reach no account, the share given the"
        " account that gold's steps name (`workflow_links.py`).",
        "- A metric with an empty denominator is shown as n/a.",
        "",
        "To reproduce: `python3 code/main.py`, then `python3 code/evaluation/main.py --intent`,"
        " `python3 code/evaluation/custom_cases.py`, `python3 code/evaluation/label_free_checks.py` and"
        " `python3 code/evaluation/build_reports.py`.",
    ])


def intent_section(requests: List[Dict[str, str]], samples: List[Dict[str, str]]) -> Tuple[str, Tuple[int, int, int]]:
    """The section, and (LLM correct, keyword correct, probes)."""
    verdicts = intent.classify_intents([probe.text for probe in PROBES])
    rows = []
    llm_correct = keyword_correct = 0
    for number, probe in enumerate(PROBES, 1):
        by_model, by_keyword = verdicts[probe.text], intent.keyword_intent(probe.text)
        llm_correct += by_model["deletion"] == probe.deletion
        keyword_correct += by_keyword["deletion"] == probe.deletion
        rows.append([
            number,
            probe.language,
            probe.note,
            "deletion" if probe.deletion else "not",
            "deletion" if by_model["deletion"] else "not",
            "deletion" if by_keyword["deletion"] else "not",
            "yes" if by_model["deletion"] == probe.deletion else "**no**",
            "yes" if by_keyword["deletion"] == probe.deletion else "**no**",
        ])
    sample_verdicts = [agent.intent_for(case["request_text"]) for case in samples]
    request_verdicts = [agent.intent_for(case["request_text"]) for case in requests]
    refused = sum(not verdict["deletion"] for verdict in request_verdicts)
    text = "\n".join([
        "## Intent classification",
        "",
        "Whether a request authorises deletion is the one decision the language model makes about a case. It was tested on"
        " %s (`code/evaluation/intent_cases.py`): non-deletion requests that contain deletion verbs, and"
        " deletion requests in languages the keyword fallback does not cover." % plural(len(PROBES), "hand-written probe"),
        "",
        table(
            ("#", "Language", "Probe", "Expected", "LLM", "Keyword", "LLM right", "Keyword right"), rows, right=(0,)
        ),
        "",
        "- LLM (%s): **%s** correct." % (llm.MODEL, fraction(llm_correct, len(PROBES))),
        "- Keyword fallback: **%s** correct." % fraction(keyword_correct, len(PROBES)),
        "- On the %s (gold has no non-deletion among them) %s are classified as deletions."
        % (plural(len(samples), "labelled sample"), fraction(sum(verdict["deletion"] for verdict in sample_verdicts), len(samples))),
        "- On the %s, %d are NOT_APPLICABLE; %s of all verdicts come from the model, and the run"
        " refuses to write output if any came from the keyword fallback."
        % (plural(len(requests), "evaluation request"), refused,
           fraction(sum(verdict["source"] == "llm" for verdict in request_verdicts), len(requests))),
    ])
    return text, (llm_correct, keyword_correct, len(PROBES))


def custom_section(results: List[custom_cases.CaseResult]) -> str:
    rows = [[result.name, result.title, "PASS" if result.passed else "**FAIL**", result.detail] for result in results]
    return "\n".join([
        "## Custom cases",
        "",
        "Each case takes a labelled sample request, changes one thing, keeps the `case_id`, and runs the agent."
        " The samples' gold answers are not used.",
        "",
        table(("Case", "What must hold", "Result", "Observed"), rows),
        "",
        "Custom cases passed: **%d of %d**." % (sum(result.passed for result in results), len(results)),
    ])


def label_free_section(checks: label_free_checks.Checks) -> str:
    na = len(checks.not_applicable)
    marked_na = sum(
        1 for links in checks.intake_links.values() if any(status == "NOT_APPLICABLE" for _, status in links)
    )
    checks_rows = [
        ["NOT_APPLICABLE verdicts linked to an account the privacy intake also marked non-deletion",
         "%d of %d" % (len(checks.agreeing), na), "all"],
        ["Steps citing evidence captured after the case's `audit_at`", len(checks.late_citations), "0"],
        ["Steps citing an id that is in no source file, activity or audit row", len(checks.unknown_citations), "0"],
        ["COMPLETE steps with an empty `evidence_id`", len(checks.empty_complete), "0"],
        ["Remediation steps without a matching verification step", len(checks.unmatched_remediation), "0"],
    ]
    intake_rows = [
        [link.case_id, link.account, link.activity_id, link.activity_time, "yes" if link.in_window else "**no**"]
        for link in checks.agreeing
    ]
    intake_rows += [[case_id, "(no intake-marked account)", "", "", ""] for case_id in checks.disagreeing]
    matrix = [[reported] + counts + [sum(counts)] for reported, counts in label_free_checks.matrix_rows(checks)]
    column_totals = [sum(counts[number] for _, counts in label_free_checks.matrix_rows(checks)) for number in range(len(STATUSES))]
    matrix.append(["**Total**"] + column_totals + [checks.cases])

    claims = []
    for reported, counts in label_free_checks.matrix_rows(checks):
        total = sum(counts)
        claims.append(
            "- The workflow reported **%s** for %s; our audit finds %d COMPLETE (%.1f%%), %d INCOMPLETE (%.1f%%)"
            " and %d NOT_APPLICABLE (%.1f%%)."
            % (reported, plural(total, "request"), counts[0], 100.0 * counts[0] / total, counts[1], 100.0 * counts[1] / total,
               counts[2], 100.0 * counts[2] / total)
        )

    lines = [
        "## Label-free checks on the %s" % plural(checks.cases, "evaluation request"),
        "",
        "Run on the agent's predictions with no gold answers. A non-zero count in any row marked \"0\" is a bug.",
        "",
        table(("Check", "Result", "Expected"), checks_rows, right=(1,)),
        "",
        "### Agreement with the privacy intake",
        "",
        "`system_activity.csv` records the privacy intake's own CLASSIFY_REQUEST activities: %s closed as"
        " non-deletion. The intent classifier never sees them. Accounts are reached from the request email through"
        " current and historical identifiers only." % plural(len(checks.intake_accounts), "account"),
        "",
        table(("Case", "Account", "Intake activity", "Intake marked at", "Inside request window"), intake_rows),
        "",
        "Seen from the other side, intake-marked accounts that belong to a request we answered NOT_APPLICABLE: **%d of %d**."
        % (marked_na, len(checks.intake_accounts)),
        "",
        "### Workflow status against our audit",
        "",
        "The workflow's `reported_status` is a claim, not proof. Rows are the claim, columns our `deletion_status`.",
        "",
        table(("Reported status",) + tuple(STATUSES) + ("Total",), matrix, right=range(1, len(STATUSES) + 2)),
        "",
    ] + claims
    return "\n".join(lines)


def evaluation_report(
    metrics: List[Tuple[str, str]],
    links: Tuple[int, int, int, int],
    requests: List[Dict[str, str]],
    samples: List[Dict[str, str]],
    results: List[custom_cases.CaseResult],
    checks: label_free_checks.Checks,
) -> str:
    current = dict(metrics)
    intent_text, _ = intent_section(requests, samples)
    parts = [
        "# Evaluation report\n\nGenerated by `python3 code/evaluation/build_reports.py`. Every number below was computed in"
        " that run from the evaluator, the custom cases and the label-free checks.",
        narrative("approach"),
        method_section(len(requests), len(samples), len(results)),
        metrics_section(metrics, links, len(samples)),
        progression_section(current),
        intent_text,
        custom_section(results),
        label_free_section(checks),
        narrative("failures"),
        narrative("limitations"),
    ]
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------- usage report

def read_jsonl(path: Path) -> List[Entry]:
    if not path.exists():
        return []
    with open(str(path), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cost_of(entry: Entry) -> float:
    """Dollars for one call, from its tokens and the price of its model."""
    price = llm.PRICES.get(entry["model"])
    if price is None:
        raise ValueError("the ledger has a call to %r, which has no price in llm.PRICES" % entry["model"])
    return (entry["input_tokens"] * price["input"] + entry["output_tokens"] * price["output"]) / 1000000.0


def totals(entries: List[Entry]) -> Dict[str, float]:
    return {
        "calls": len(entries),
        "input": sum(entry["input_tokens"] for entry in entries),
        "output": sum(entry["output_tokens"] for entry in entries),
        "cost": sum(cost_of(entry) for entry in entries),
    }


def usage_row(label: str, entries: List[Entry]) -> List[str]:
    t = totals(entries)
    return [label, "{:,}".format(t["calls"]), "{:,}".format(t["input"]), "{:,}".format(t["output"]),
            "{:,}".format(t["input"] + t["output"]), "$%.4f" % t["cost"]]


USAGE_HEADERS = ("", "Calls", "Input tokens", "Output tokens", "Total tokens", "Est. cost (USD)")

UNTAGGED = "Before run tagging (no `run_id`)"
ORIGINS = {
    "evaluation": "Evaluation scripts",
    "main": "`main` runs other than the final run",
    "explain": "`main.py --explain`",
    "unspecified": "No entrypoint set",
}


def usage_table(entries: List[Entry], models: Sequence[str] = (), purposes: Sequence[str] = ()) -> str:
    """Rows per model and purpose, a total per model, and an overall row when there are several models.

    models and purposes name the rows to show even when they made no call; by default only those present.
    """
    models = models or sorted({entry["model"] for entry in entries})
    rows = []
    for model in models:
        mine = [entry for entry in entries if entry["model"] == model]
        for purpose in purposes or sorted({entry["purpose"] for entry in mine}):
            rows.append(usage_row("%s / %s" % (model, purpose), [e for e in mine if e["purpose"] == purpose]))
        rows.append(usage_row("**%s total**" % model, mine))
    if len(models) > 1:
        rows.append(usage_row("**All models**", entries))
    return table(("Model / purpose",) + USAGE_HEADERS[1:], rows, right=range(1, len(USAGE_HEADERS)))


def origin(entry: Entry) -> str:
    if not entry.get("run_id"):
        return UNTAGGED
    return ORIGINS.get(entry["entrypoint"], "`%s`" % entry["entrypoint"])


def origin_table(entries: List[Entry]) -> str:
    """Where the calls came from: one row per origin, then the total."""
    def runs(mine: List[Entry]) -> int:
        return len({entry["run_id"] for entry in mine if entry.get("run_id")})

    labels = sorted({origin(entry) for entry in entries}, key=lambda label: (label != UNTAGGED, label))
    rows = []
    for label in labels:
        mine = [entry for entry in entries if origin(entry) == label]
        rows.append([label, runs(mine) if label != UNTAGGED else "n/a"] + usage_row(label, mine)[1:])
    rows.append(["**All development and evaluation calls**", runs(entries)] + usage_row("", entries)[1:])
    return table(("Origin", "Runs with calls") + USAGE_HEADERS[1:], rows, right=range(1, len(USAGE_HEADERS) + 1))


def final_run_section(
    final: Optional[Entry],
    entries: List[Entry],
    replays: List[Entry],
    models: Sequence[str],
    purposes: Sequence[str],
    requests: int,
    cached: int,
) -> str:
    """The final run, from its run record and the ledger entries tagged with its run id.

    It is the latest main.py run that made API calls (the latest cold run). replays are the main runs
    after it, which made none.
    """
    if final is None:
        return "\n".join([
            "## Final run",
            "",
            "No finished run of `python3 code/main.py` is recorded in `code/cache/llm_runs.jsonl` yet, so there is no final"
            " run to describe. Run it, then rebuild this report.",
        ])
    count = final["requests"]
    t = totals(entries)
    tokens = t["input"] + t["output"]
    if t["calls"]:
        lines = [
            "## Final run",
            "",
            "The final run is the latest run of `python3 code/main.py` that made API calls, the latest cold run: run `%s`,"
            " finished %s UTC, answering %s." % (final["run_id"], final["finished_at"], plural(count, "request")),
        ]
        if count != requests:
            lines.append("It covered %s, not the %d in `dataset/deletion_requests.csv`." % (plural(count, "request"), requests))
        lines += [
            "",
            "It made **%s** and read %s from the response cache."
            % (plural(t["calls"], "API call"), plural(final["cache_hits"], "answer")),
        ]
    else:
        lines = [
            "## Final run",
            "",
            "No run of `python3 code/main.py` has made API calls, so the final run is the latest one: run `%s`, finished %s"
            " UTC, answering %s." % (final["run_id"], final["finished_at"], plural(count, "request")),
            "",
            "**This run made 0 API calls.** All %s it needed (intent verdicts and screenshot readings) were read from the"
            " response cache `code/cache/llm_cache.json` (%s), so it cost $0.0000. The cache was filled by the development"
            " calls below: this run repeats those answers for free, it does not make them unnecessary."
            % (plural(final["cache_hits"], "model answer"), plural(cached, "entry", "entries")),
        ]
    lines += [
        "",
        usage_table(entries, models, purposes),
        "",
        "Per request, over the %s of this run:" % plural(count, "request"),
        "",
        table(("Measure", "Value"), [
            ["Total tokens", "{:,}".format(tokens)],
            ["Average tokens per request", "{:,.0f}".format(tokens / count) if count else "n/a"],
            ["Estimated total cost", "$%.4f" % t["cost"]],
            ["Estimated cost per request", "$%.6f" % (t["cost"] / count) if count else "n/a"],
        ], right=(1,)),
    ]
    if replays:
        lines += [
            "",
            "### Cache replays",
            "",
            "Later runs of `python3 code/main.py` made no API calls: every answer came from the response cache. They do not"
            " replace the final run.",
            "",
            table(("Run", "Requests", "Cache hits", "API calls"),
                  [["`%s`" % run["run_id"], run["requests"], run["cache_hits"], 0] for run in replays], right=(1, 2, 3)),
        ]
    return "\n".join(lines)


def usage_report(entries: List[Entry], runs: List[Entry], requests: int, new_calls: int) -> str:
    models = sorted({entry["model"] for entry in entries} | {llm.MODEL})
    purposes = sorted({entry["purpose"] for entry in entries})
    cached = len(json.loads(llm.CACHE_PATH.read_text(encoding="utf-8")))

    def of_run(run: Entry) -> List[Entry]:
        return [entry for entry in entries if (entry.get("run_id"), entry.get("entrypoint")) == (run["run_id"], run["entrypoint"])]

    mains = [run for run in runs if run["entrypoint"] == "main"]  # the run log is append-only: oldest first
    cold = [run for run in mains if of_run(run)]
    final = cold[-1] if cold else (mains[-1] if mains else None)
    position = mains.index(final) if final is not None else len(mains)
    replays = mains[position + 1:]  # after a cold final run these made no calls; none if the final run is the latest
    earlier = [run for run in mains[:position] if not of_run(run)]

    final_entries = of_run(final) if final is not None else []
    in_final = {id(entry) for entry in final_entries}
    development = [entry for entry in entries if id(entry) not in in_final]
    whole = totals(entries)
    spent = totals(development)
    recorded = sum(entry["cost_usd"] for entry in entries)
    untagged = sum(1 for entry in entries if not entry.get("run_id"))
    price_rows = [[model, "$%.2f" % llm.PRICES[model]["input"], "$%.2f" % llm.PRICES[model]["output"]] for model in models]

    notes = [
        "- The final run is the latest finished `python3 code/main.py` run that made API calls, the latest cold run; if no run"
        " did, it is the latest run. Only calls tagged with its run id count towards it. Later runs served from the cache are"
        " listed as replays, and a `--explain`, evaluation or unfinished run never replaces it.",
        "- Calls made before run tagging existed (%d) carry no `run_id` and count as development." % untagged,
    ]
    if earlier:
        notes.append(
            "- `main` runs before the final run that made no API calls: %s. They ran on a cache that the development calls had"
            " filled." % ", ".join("`%s`" % run["run_id"] for run in earlier)
        )
    notes.append(
        "- Generating these reports was itself an evaluation run: it ran the samples, the %s, the custom cases and the probes"
        " and made **%s**." % (plural(requests, "evaluation request"), plural(new_calls, "new API call"))
    )
    if not cold:
        notes.append(
            "- No `main` run has made calls, so the final run's cost is zero only because the cache was filled earlier."
            " Repeating it with an empty cache would redo that work; the development total above is an upper bound on that"
            " cost (about $%.6f per request over %s), not a measurement of it."
            % (spent["cost"] / requests if requests else 0.0, plural(requests, "request"))
        )

    lines = [
        "# Usage report",
        "",
        "Generated by `python3 code/evaluation/build_reports.py` from the usage ledger `code/cache/llm_usage.jsonl` and the"
        " run log `code/cache/llm_runs.jsonl`, both written by `code/llm.py`. The ledger has a line per real API call,"
        " tagged with the `run_id` and `entrypoint` of the script that made it. The run log has a line per finished run of"
        " `python3 code/main.py`, so a run served entirely from the cache is on record too.",
        "",
        "## Models",
        "",
        "- Provider: Anthropic.",
        "- Model%s: %s." % ("s" if len(models) > 1 else "", ", ".join("`%s`" % model for model in models)),
        "- Used for: `intent` (does the request authorise deletion?) and `vision` (reading screenshots into evidence)."
        " Everything else, including every status and step, is decided by deterministic code.",
        "- Sampling settings: none. The model rejects `temperature` (see the comment in `code/llm.py`), so repeatability"
        " comes from the response cache.",
        "",
        final_run_section(final, final_entries, replays, models, purposes, requests, cached),
        "",
        "## Development and evaluation calls",
        "",
        "Every other call in the ledger, so that nothing is left out: %s, $%.4f. Together with the final run they add up to"
        " the whole ledger of %s, $%.4f."
        % (plural(spent["calls"], "call"), spent["cost"], plural(whole["calls"], "call"), whole["cost"]),
        "",
        usage_table(development, models, purposes) if development else "No other calls.",
        "",
        origin_table(development) if development else "",
        "",
        "## What these numbers cover",
        "",
    ] + notes + [
        "",
        "## Cost estimate",
        "",
        "Cost is tokens times the price below, summed per call. Over the whole ledger it %s the cost recorded there ($%.4f)."
        % ("matches" if abs(recorded - whole["cost"]) < 0.0001 else "**differs from**", recorded),
        "",
        table(("Model", "Input, USD per million tokens", "Output, USD per million tokens"), price_rows, right=(1, 2)),
        "",
        "Prices are for prompts up to 100K tokens. Source: %s." % llm.PRICES_SOURCE,
        "Spend is capped by `code/llm.py`: new calls are refused once the ledger total reaches $%.2f"
        " (`LLM_SPEND_CAP_USD`)." % llm.DEFAULT_SPEND_CAP_USD,
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main

def write(path: Path, text: str) -> None:
    with open(str(path), "w", newline="\n", encoding="utf-8") as handle:
        handle.write(text)


def main() -> int:
    llm.start_run("evaluation")
    sample_evaluator = load_sample_evaluator()
    requests = pipeline.read_requests(sample_evaluator.EVALUATION_REQUESTS)
    samples = pipeline.read_requests(sample_evaluator.SAMPLE_REQUESTS)
    calls_before = len(read_jsonl(llm.LEDGER_PATH))

    try:
        sample_rows = pipeline.run_agent(samples)
        request_rows = pipeline.run_agent(requests)
    except (pipeline.KeywordFallbackError, pipeline.MissingVisionError) as exc:
        sys.stderr.write("ERROR: %s\nFix the cause (API key, spend cap, network) and re-run.\n" % exc)
        return 1

    gold = sample_evaluator.load_cases(sample_evaluator.GOLD)
    predictions = {case_id: (status, steps) for case_id, status, steps in sample_rows}
    metrics = sample_evaluator.compute_metrics(predictions, gold)
    links = sample_evaluator.workflow_link_counts(gold)
    results = custom_cases.run_cases()
    checks = label_free_checks.run_checks(requests, request_rows)

    report = evaluation_report(metrics, links, requests, samples, results, checks)
    entries, runs = read_jsonl(llm.LEDGER_PATH), read_jsonl(llm.RUNS_PATH)
    write(EVALUATION_REPORT, report)
    write(USAGE_REPORT, usage_report(entries, runs, len(requests), len(entries) - calls_before))

    print("Wrote %s and %s" % (EVALUATION_REPORT, USAGE_REPORT))
    failed = [result.name for result in results if not result.passed]
    if failed or checks.violations():
        print("WARNING: custom cases failed: %s; label-free violations: %d" % (", ".join(failed) or "none", checks.violations()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
