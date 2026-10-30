#!/usr/bin/env python3
"""Custom evaluation cases: metamorphic tests derived from the public samples.

Each case takes a sample request, changes exactly one thing (keeping the case_id and every
other field) and runs it through predict(). Changing one thing must change only what it
should: a non-deletion text must end in NOT_APPLICABLE; an old email or another language
must not change the answer; an earlier audit_at must stop later evidence from being cited.
The samples' gold answers are not used.

Usage: python3 code/evaluation/custom_cases.py
Exits 1 if any case fails. A failure reports what was observed; the cases are not adjusted.
"""
import sys
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Tuple

EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import agent  # noqa: E402
import llm  # noqa: E402
from identity import resolve_identity  # noqa: E402
import main as pipeline  # noqa: E402  (code/main.py: CODE_DIR is first on the path)

from label_free_checks import cited_late, evidence_times  # noqa: E402
from wording import plural  # noqa: E402

SAMPLE_REQUESTS = REPO_ROOT / "dataset" / "sample_deletion_requests.csv"

Case = Dict[str, str]
Prediction = Tuple[str, List[dict]]


class CaseResult(NamedTuple):
    name: str
    title: str  # what was changed and what must follow
    passed: bool
    detail: str  # what was observed


def derive(case: Case, **changes: str) -> Case:
    """The same request with some fields replaced; case_id stays."""
    derived = dict(case)
    derived.update(changes)
    return derived


def _summary(prediction: Prediction) -> str:
    status, steps = prediction
    return "%s with %s" % (status, plural(len(steps), "step"))


def _cited(steps: List[dict]) -> List[str]:
    return sorted({step["evidence_id"] for step in steps if step["evidence_id"]})


def _model_verdict(text: str) -> Tuple[bool, str]:
    """(answered by the model, 'DELETION/NOT a deletion (source)'): a keyword verdict would test the fallback."""
    verdict = agent.intent_for(text)
    return verdict["source"] == "llm", "%s (%s)" % ("deletion" if verdict["deletion"] else "not a deletion", verdict["source"])


def c1_not_a_deletion(cases: Dict[str, Case]) -> CaseResult:
    text = "Please correct the phone number on my account. I want to keep using it."
    derived = derive(cases["case_05"], request_text=text)
    prediction = agent.predict(derived)
    from_model, verdict = _model_verdict(text)
    return CaseResult(
        "C1",
        "not a deletion: case_05 asking to correct a phone number is NOT_APPLICABLE with no steps",
        prediction == ("NOT_APPLICABLE", []) and from_model,
        "intent: %s; result: %s" % (verdict, _summary(prediction)),
    )


def c2_old_email(cases: Dict[str, Case]) -> CaseResult:
    old_email = "minjun.kim90@gmail.com"  # case_12's address before 2026-05-18 (user_audit_log, audit_282)
    base = cases["case_12"]
    expected, got = agent.predict(base), agent.predict(derive(base, email=old_email))
    users = ", ".join(resolve_identity(old_email).user_ids) or "(none)"
    return CaseResult(
        "C2",
        "old email: case_12 asked from its historical address gives the same output as from the current one",
        got == expected,
        "%s resolves to %s; current email: %s; old email: %s" % (old_email, users, _summary(expected), _summary(got)),
    )


def c3_time_cutoff(cases: Dict[str, Case]) -> CaseResult:
    late_file = "evidence_54"  # failover replica of customer_accounts, captured 2026-09-02T23:00:00Z
    audit_at = "2026-09-02T22:00:00Z"
    base = cases["case_05"]
    base_prediction = agent.predict(base)
    prediction = agent.predict(derive(base, audit_at=audit_at))
    late, unknown = cited_late(audit_at, prediction[1], evidence_times())
    cited_before = late_file in _cited(base_prediction[1])
    ok = cited_before and late_file not in _cited(prediction[1]) and not late and not unknown
    detail = "audit_at %s: %s, cites %s; original audit_at %s: %s, %s %s" % (
        audit_at, _summary(prediction), ", ".join(_cited(prediction[1])) or "nothing",
        base["audit_at"], _summary(base_prediction), "cites" if cited_before else "does not cite", late_file,
    )
    if not cited_before:
        detail += " (so the case cannot show a cut-off)"
    return CaseResult(
        "C3",
        "time cut-off: case_05 audited before %s was captured does not cite it, or anything later" % late_file,
        ok,
        detail,
    )


def c4_other_language(cases: Dict[str, Case]) -> CaseResult:
    text = "Bitte löschen Sie mein Konto und alle damit verbundenen personenbezogenen Daten dauerhaft."
    base = cases["case_01"]
    expected, got = agent.predict(base), agent.predict(derive(base, request_text=text))
    from_model, verdict = _model_verdict(text)
    return CaseResult(
        "C4",
        "other language: case_01 requested in German gives the same output as in English",
        got == expected and from_model,
        "intent: %s; English: %s; German: %s" % (verdict, _summary(expected), _summary(got)),
    )


CASES = (c1_not_a_deletion, c2_old_email, c3_time_cutoff, c4_other_language)  # type: Tuple[Callable[[Dict[str, Case]], CaseResult], ...]


def run_cases() -> List[CaseResult]:
    cases = {case["case_id"]: case for case in pipeline.read_requests(SAMPLE_REQUESTS)}
    return [check(cases) for check in CASES]


def render_text(results: List[CaseResult]) -> str:
    lines = []
    for result in results:
        lines.append("%s %s  %s" % (result.name, "PASS" if result.passed else "FAIL", result.title))
        lines.append("    %s" % result.detail)
    lines.append("")
    lines.append("%d/%d custom cases passed" % (sum(result.passed for result in results), len(results)))
    return "\n".join(lines)


def main() -> int:
    llm.start_run("evaluation")
    results = run_cases()
    print(render_text(results))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
