"""Deletion-audit agent: rule-based case engine.

For one request: resolve the subject's identity, link the resources that are theirs,
decide each resource's state at audit_at from admissible evidence, apply
deletion_rules.csv, and emit ordered FINDING / REMEDIATION / VERIFICATION steps.

audit_case() returns the full audit trail, with a reason (rule, state, evidence) for every
step; main.py --explain prints it. predict() returns only what output.csv needs.

Before any of that, intent.py decides whether the request text asks for deletion at all;
a request that does not is NOT_APPLICABLE with no steps. prepare() classifies a whole
request list in one go (batched, cached) so predict() only looks verdicts up.

A request whose identifiers reach no account_service resource may get its primary account
from the deletion worker's activity stream instead (workflow_links.py).

State is decided only by evidence observed inside the audit window, requested_at to
audit_at, so an older screenshot cannot create a finding. A backup or data flow whose
evidence does not state that it is blocked or disabled is treated as active: unproven
safety is unsafe.
"""
import csv
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evidence
import workflow_links
from identity import Identity, resolve_identity
from intent import classify_intents
from validate import ALLOWED_OPERATIONS, OPERATION_ORDER, PHASES

RULES_PATH = evidence.DATASET_DIR / "deletion_rules.csv"
ACTIVITY_PATH = evidence.DATASET_DIR / "system_activity.csv"

SYSTEM_DEFAULT_CATEGORY = {
    "account_service": "USER_PROFILE",
    "search_service": "SEARCH_INDEX_ENTRY",
    "analytics_platform": "ANALYTICS_PROFILE",
    "support_platform": "SUPPORT_MESSAGE",
    "export_service": "USER_EXPORT",
    "backup_service": "BACKUP_ARCHIVE",
    "integration_service": "DATA_FLOW",
    "security_service": "SECURITY_AUDIT_LOG",
    "identity_service": "ACCOUNT_TOMBSTONE",
    "legal_service": "LEGAL_HOLD_RECORD",
    "cache_service": "CACHE_ENTRY",
    "email_delivery_service": "EMAIL_DELIVERY_LOG",
    "marketing_platform": "MARKETING_PROFILE",
}

# Categories whose treatment concerns a process rather than the record, overriding
# required_action: BACKUP_ARCHIVE is RETAIN in the rules, and DATA_FLOW has no rule.
PROCESS_POLICIES = {"BACKUP_ARCHIVE": "RESTORE", "DATA_FLOW": "FLOW"}

# The only states that prove a process cannot recreate data; any other present state,
# including PRESENT with no explicit restore or enabled flag, counts as ACTIVE.
SAFE_PROCESS_STATES = {"RESTORE": ("ABSENT", "BLOCKED"), "FLOW": ("ABSENT", "DISABLED")}

# policy -> (finding, remediation, verification) operations.
OPERATIONS = {
    "DELETE": ("REPORT_REMAINING_RECORD", "DELETE_RECORD", "VERIFY_RECORD_ABSENT"),
    "REDACT": ("REPORT_EXPOSED_PII", "REDACT_RECORD", "VERIFY_RECORD_REDACTED"),
    "ANONYMIZE": ("REPORT_EXPOSED_PII", "ANONYMIZE_RECORD", "VERIFY_RECORD_ANONYMIZED"),
    "RETAIN": (None, None, "VERIFY_RECORD_PRESENT"),
    "RESTORE": ("REPORT_RESTORE_RISK", "BLOCK_RESTORE", "VERIFY_RESTORE_BLOCKED"),
    "FLOW": ("REPORT_ACTIVE_DATA_FLOW", "DISABLE_DATA_FLOW", "VERIFY_DATA_FLOW_DISABLED"),
}

NAME_FIELDS = {"name", "full_name", "display_name", "requester_name"}
LINKAGE_FIELDS = {"user_id", "customer_reference"}  # identifying only for ANONYMIZE categories

Sighting = Dict[str, Any]


def _leaf(key: str) -> str:
    return key.rsplit(".", 1)[-1]


def _events(sighting: Sighting) -> List[str]:
    found = [sighting["context"].get("event"), sighting["fields"].get("event")]
    return [event for event in found if isinstance(event, str)]


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ("", "null", "none")
    return True


# ---------------------------------------------------------------- category hints

def _attachment_signal(sighting: Sighting, file_format: str) -> bool:
    context = sighting["context"]
    return (
        "attachment" in (context.get("collection") or "")
        or any("attachment" in event for event in _events(sighting))
        or any(_leaf(key) == "file_name" for key in sighting["fields"])
        or file_format == "pdf"
    )


def _contact_signal(sighting: Sighting, file_format: str) -> bool:
    return any("contact_query" in event for event in _events(sighting)) or any(
        value == "legacy_contact_index" for value in sighting["fields"].values()
    )


def _marketing_signal(sighting: Sighting, file_format: str) -> bool:
    return any("contact_lookup" in event for event in _events(sighting)) or any(
        _leaf(key) == "email_opt_in" for key in sighting["fields"]
    )


_SUPPORT_ATTACHMENTS = re.compile(r"support[-_ /]?attachments?", re.IGNORECASE)


def _support_location_signal(sighting: Sighting, file_format: str) -> bool:
    return any(isinstance(value, str) and _SUPPORT_ATTACHMENTS.search(value) for value in sighting["fields"].values())


# Semantic category hints, checked in order after record_type and before the system default.
# (category, system the hint applies to or None for any, signal on any admissible sighting)
CATEGORY_HINTS = (
    # Support records that are files: attachment collection or event, file_name, or a PDF.
    ("SUPPORT_ATTACHMENT", "support_platform", _attachment_signal),
    # Contact lookups against the legacy contact index hold a secondary identifier.
    ("SECONDARY_IDENTIFIER", None, _contact_signal),
    # Analytics records used for contact lookups or opt-in tracking are marketing profiles.
    ("MARKETING_PROFILE", "analytics_platform", _marketing_signal),
    # Stored objects in a support-attachments location are support attachments.
    ("SUPPORT_ATTACHMENT", "object_storage", _support_location_signal),
)

# Category hints from the resource's own LINK_RESOURCE_IDENTITY activities, checked before
# CATEGORY_HINTS: (category, system, phrase in the activity details, case-insensitive).
LINK_HINTS = (("SUPPORT_ATTACHMENT", "object_storage", "attachment catalog"),)


# ---------------------------------------------------------------- indexes

class _Index:
    """Everything built once per process: evidence, coverage, rules and link activities."""

    def __init__(self) -> None:
        manifest = evidence.read_manifest()
        self.file_captured = {entry["file_id"]: entry["captured_at"] for entry in manifest}
        self.file_format = {entry["file_id"]: Path(entry["file_path"]).suffix.lstrip(".").lower() for entry in manifest}

        self.by_resource = defaultdict(list)  # type: Dict[str, List[Tuple[int, Sighting]]]
        self.by_identifier = defaultdict(list)  # type: Dict[str, List[Sighting]]
        for order, sighting in enumerate(evidence.load_sightings()):
            self.by_resource[sighting["resource_id"]].append((order, sighting))
            for identifier in sighting["identifiers"]:
                self.by_identifier[identifier].append(sighting)

        self.coverage_by_system = defaultdict(list)  # type: Dict[str, List[Dict[str, Any]]]
        for entry in evidence.load_coverage():
            table = entry.get("context", {}).get("table") or ""
            if "replica" in table:
                continue  # a replica's listing says nothing about the primary copy
            self.coverage_by_system[entry["system_id"]].append(dict(entry, listed=frozenset(entry["resource_ids"])))

        with open(str(RULES_PATH), newline="", encoding="utf-8-sig") as handle:
            self.rules = {row["data_category"]: row for row in csv.DictReader(handle)}
        for row in self.rules.values():
            if row["required_action"] not in OPERATIONS:
                raise ValueError("%s: unknown required_action %r" % (row["rule_id"], row["required_action"]))

        self.link_activities = []  # type: List[Tuple[Dict[str, str], frozenset]]
        self.link_rows = defaultdict(list)  # type: Dict[str, List[Dict[str, str]]]
        with open(str(ACTIVITY_PATH), newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["event_type"] == "LINK_RESOURCE_IDENTITY":
                    self.link_activities.append((row, frozenset(evidence.extract_identifiers([row["details"]]))))
                    self.link_rows[row["resource_id"]].append(row)


@lru_cache(maxsize=None)
def _index() -> _Index:
    return _Index()


# ---------------------------------------------------------------- per-resource assessment

def _category(
    resource_id: str, system_id: str, sightings: List[Tuple[int, Sighting]], index: _Index
) -> Tuple[Optional[str], str]:
    typed = [(s["observed_at"], order, s) for order, s in sightings if _has_value(s["fields"].get("record_type"))]
    if typed:
        latest = max(typed, key=lambda item: item[:2])[2]
        return latest["fields"]["record_type"], "record_type in %s" % latest["evidence_id"]
    for category, hint_system, phrase in LINK_HINTS:
        if hint_system != system_id:
            continue
        for row in index.link_rows.get(resource_id, ()):
            if phrase in row["details"].lower():
                return category, "%s in %s" % (phrase, row["activity_id"])
    for category, hint_system, signal in CATEGORY_HINTS:
        if hint_system is not None and hint_system != system_id:
            continue
        for _, s in sightings:
            if signal(s, index.file_format[s["evidence_id"]]):
                return category, "%s hint from %s" % (signal.__name__.strip("_"), s["evidence_id"])
    default = SYSTEM_DEFAULT_CATEGORY.get(system_id)
    return default, "system default" if default else "no category for this system"


def _policy(category: str, index: _Index) -> Tuple[Optional[str], Optional[str]]:
    rule = index.rules.get(category)
    rule_id = rule["rule_id"] if rule else None
    if category in PROCESS_POLICIES:
        return PROCESS_POLICIES[category], rule_id
    return (rule["required_action"], rule_id) if rule else (None, None)


def _exposure(sighting: Sighting, anonymize: bool) -> str:
    """IDENTIFIABLE, IDENTIFIERS_REMOVED (identifier fields exist and are all null) or
    NOT_KNOWN_REMOVED (nothing shows the identifiers were removed)."""
    fields = sighting["fields"]
    if sighting["state"] == "IDENTIFIABLE":
        return "IDENTIFIABLE"
    scanned = any(event.endswith("_pii_scan") or event == "analytics_identity_scan" for event in _events(sighting))
    if scanned and _has_value(fields.get("matched_value")):
        return "IDENTIFIABLE"
    if _has_value(fields.get("direct_identifiers")):
        return "IDENTIFIABLE"
    identifying = []
    for key, value in fields.items():
        leaf = _leaf(key)
        if (
            leaf in NAME_FIELDS
            or leaf in ("email", "email_hash", "phone")
            or leaf.endswith(("_email", "_email_hash", "_phone"))
            or (anonymize and leaf in LINKAGE_FIELDS)
        ):
            identifying.append(value)
    if any(_has_value(value) for value in identifying):
        return "IDENTIFIABLE"
    return "IDENTIFIERS_REMOVED" if identifying else "NOT_KNOWN_REMOVED"


def _reference_only(sighting: Sighting, identity: Identity) -> bool:
    """True if the only identifier the record shows is the subject's own user reference."""
    shown = set(sighting["identifiers"])
    return bool(shown) and shown <= {user_id.lower() for user_id in identity.user_ids}


def _judge(policy: str, copy: Dict[str, Any], category: str, identity: Identity) -> Tuple[bool, str]:
    """(violates, assessed state) for one copy under the policy."""
    state = copy["state"]
    if policy == "DELETE":
        return state != "ABSENT", state
    if policy in ("REDACT", "ANONYMIZE"):
        if state in ("ABSENT", "REDACTED", "ANONYMIZED"):
            return False, state
        if state == "IDENTIFIABLE":
            return True, state
        exposure = _exposure(copy["sighting"], policy == "ANONYMIZE")
        # The policy's own tombstone: the stable user reference and nothing else.
        if category == "ACCOUNT_TOMBSTONE" and exposure == "NOT_KNOWN_REMOVED" and _reference_only(copy["sighting"], identity):
            return False, "REDACTED"
        return exposure != "IDENTIFIERS_REMOVED", exposure
    if policy == "RETAIN":
        return False, state
    # RESTORE, FLOW: unproven safety is unsafe.
    if state in SAFE_PROCESS_STATES[policy] or state == "ACTIVE":
        return state == "ACTIVE", state
    return True, "ACTIVE (no explicit %s state)" % ("restore" if policy == "RESTORE" else "enabled")


def _copies(
    resource_id: str,
    system_id: str,
    sightings: List[Tuple[int, Sighting]],
    requested_at: str,
    audit_at: str,
    index: _Index,
) -> List[Dict[str, Any]]:
    """State per copy: the latest in-window explicit sighting decides; in-window coverage
    proves the primary copy ABSENT only when it has no explicit sighting."""
    groups = defaultdict(list)  # type: Dict[str, List[Tuple[int, Sighting]]]
    for order, s in sightings:
        table = s["context"].get("table") or ""
        groups[table if "replica" in table else "primary"].append((order, s))

    copies = []
    for name in sorted(groups, key=lambda name: (name != "primary", name)):
        order, latest = max(groups[name], key=lambda item: (item[1]["observed_at"], item[1]["captured_at"], item[0]))
        copies.append({
            "copy": name,
            "state": latest["state"],
            "evidence_id": latest["evidence_id"],
            "observed_at": latest["observed_at"],
            "source": "sighting",
            "sighting": latest,
        })

    if "primary" not in groups:
        omitting = [
            entry
            for entry in index.coverage_by_system.get(system_id, ())
            if requested_at <= entry["observed_at"] <= audit_at
            and index.file_captured[entry["evidence_id"]] <= audit_at
            and resource_id not in entry["listed"]
        ]
        if omitting:
            latest = max(omitting, key=lambda entry: (entry["observed_at"], index.file_captured[entry["evidence_id"]]))
            copies.insert(0, {
                "copy": "primary",
                "state": "ABSENT",
                "evidence_id": latest["evidence_id"],
                "observed_at": latest["observed_at"],
                "source": "coverage (%s)" % ", ".join("%s=%s" % item for item in sorted(latest["context"].items())),
                "sighting": None,
            })
    return copies


def _identity_note(sightings: List[Tuple[int, Sighting]], identity: Identity) -> Optional[str]:
    """'identity mismatch' when the evidence shows direct identifiers and none are the subject's.
    user_<n> references are stable references, not direct identifiers."""
    shown = {i for _, s in sightings for i in s["identifiers"] if not i.startswith("user_")}
    if shown and not shown & identity.identifiers:
        return "identity mismatch: the evidence shows %d direct identifier(s), none of them the subject's" % len(shown)
    return None


def _assess(
    resource_id: str, linked_by: List[str], case: Dict[str, str], identity: Identity, index: _Index
) -> Dict[str, Any]:
    resource = {"resource_id": resource_id, "linked_by": linked_by, "system_id": evidence.system_for(resource_id)}
    system_id = resource["system_id"]
    if system_id is None:
        resource["skipped"] = "unknown resource prefix"
        return resource
    requested_at, audit_at = case["requested_at"], case["audit_at"]
    admissible = [
        (order, s)
        for order, s in index.by_resource.get(resource_id, ())
        if s["captured_at"] <= audit_at and s["observed_at"] <= audit_at
    ]
    category, category_source = _category(resource_id, system_id, admissible, index)
    resource.update(category=category, category_source=category_source)
    policy, rule_id = _policy(category, index) if category else (None, None)
    resource.update(policy=policy, rule_id=rule_id)
    if policy is None:
        resource["skipped"] = "no deletion rule for category %s" % category
        return resource

    in_window = [(order, s) for order, s in admissible if s["observed_at"] >= requested_at]
    copies = _copies(resource_id, system_id, in_window, requested_at, audit_at, index)
    if not copies:
        reason = "no evidence in the audit window"
        if admissible:
            latest = max((s for _, s in admissible), key=lambda s: s["observed_at"])
            reason += " (latest earlier sighting: %s @ %s)" % (latest["evidence_id"], latest["observed_at"])
        resource["skipped"] = reason
        return resource
    note = _identity_note(in_window, identity)
    if note:
        resource["note"] = note
    for copy in copies:
        copy["violates"], copy["assessed"] = _judge(policy, copy, category, identity)
    violating = [copy for copy in copies if copy["violates"]]
    resource["copies"] = copies
    resource["decided"] = violating[0] if violating else copies[0]
    resource["violates"] = bool(violating)
    return resource


# ---------------------------------------------------------------- case

_intents = {}  # type: Dict[str, Dict[str, Any]]  # request_text -> verdict, filled by prepare()


def prepare(cases: List[Dict[str, str]]) -> None:
    """Classify the intent of every request in cases once; call before predict() on a list."""
    missing = [text for text in dict.fromkeys(case["request_text"] for case in cases) if text not in _intents]
    if missing:
        _intents.update(classify_intents(missing))


def intent_for(text: str) -> Dict[str, Any]:
    if text not in _intents:
        _intents.update(classify_intents([text]))  # not prepared: classify this one alone
    return _intents[text]


def _link_by_identity(identity: Identity, index: _Index) -> Dict[str, List[str]]:
    """resource_id -> reasons, for every resource a sighting or link activity ties to the identity.

    Linking uses every sighting and link activity: which resources belong to the subject is
    not time-bound. Only the state and the cited evidence are limited to audit_at.
    """
    linked = {}  # type: Dict[str, List[str]]
    for identifier in sorted(identity.identifiers):
        for s in index.by_identifier.get(identifier, ()):
            reason = "%s in %s" % (identifier, s["evidence_id"])
            reasons = linked.setdefault(s["resource_id"], [])
            if reason not in reasons:
                reasons.append(reason)
    for activity, identifiers in index.link_activities:
        if identifiers & identity.identifiers:
            linked.setdefault(activity["resource_id"], []).append("%s (LINK_RESOURCE_IDENTITY)" % activity["activity_id"])
    return linked


def _has_account(linked: Dict[str, List[str]]) -> bool:
    return any(evidence.system_for(resource_id) == "account_service" for resource_id in linked)


@lru_cache(maxsize=None)
def workflow_assignment() -> workflow_links.Assignment:
    """Workflow-linked primary accounts for every deletion request in both request files."""
    index = _index()
    pool = workflow_links.read_pool()
    prepare(pool)  # cached per text; the evaluator and main.py classify these anyway
    deletions = [case for case in pool if intent_for(case["request_text"])["deletion"]]
    owned = workflow_links.owned_accounts(
        (resource_id, frozenset(s["identifiers"])) for resource_id, entries in index.by_resource.items() for _, s in entries
    )
    return workflow_links.assign(
        deletions, lambda case: not _has_account(_link_by_identity(resolve_identity(case["email"]), index)), owned
    )


def step_sort_key(step: Dict[str, Any]) -> Tuple[int, int, str, str]:
    """Phase, then the Allowed Operations table order, then system_id, then resource_id."""
    return (PHASES.index(step["phase"]), OPERATION_ORDER[step["operation"]], step["system_id"], step["resource_id"])


def audit_case(case: Dict[str, str]) -> Dict[str, Any]:
    intent = intent_for(case["request_text"])
    if not intent["deletion"]:
        return {"case": case, "intent": intent, "identity": None, "resources": [], "status": "NOT_APPLICABLE", "steps": []}

    index = _index()
    identity = resolve_identity(case["email"])

    linked = _link_by_identity(identity, index)
    if not _has_account(linked):
        link = workflow_assignment().links.get(case["case_id"])
        if link is not None:
            linked.setdefault(link.resource_id, []).append(link.describe())

    resources = [_assess(resource_id, linked[resource_id], case, identity, index) for resource_id in sorted(linked)]
    assessed = [resource for resource in resources if "skipped" not in resource]
    status = "INCOMPLETE" if any(resource["violates"] for resource in assessed) else "COMPLETE"

    entries = []  # (step, reason)
    for resource in assessed:
        finding, remediation, verification = OPERATIONS[resource["policy"]]
        decided = resource["decided"]
        reason = {
            "rule_id": resource["rule_id"],
            "policy": resource["policy"],
            "copy": decided["copy"],
            "state": decided["assessed"],
            "evidence": decided["evidence_id"],
            "note": resource.get("note"),
        }
        planned = []  # type: List[Tuple[str, str]]
        if resource["violates"]:
            planned += [(finding, decided["evidence_id"]), (remediation, "")]
        planned.append((verification, decided["evidence_id"] if status == "COMPLETE" else ""))
        for operation, evidence_id in planned:
            phase, state = ALLOWED_OPERATIONS[operation]
            step = {
                "step_no": 0,
                "phase": phase,
                "system_id": resource["system_id"],
                "resource_id": resource["resource_id"],
                "operation": operation,
                "state": state,
                "evidence_id": evidence_id,
            }
            entries.append((step, reason))

    entries.sort(key=lambda entry: step_sort_key(entry[0]))
    for number, (step, _) in enumerate(entries, 1):
        step["step_no"] = number

    return {
        "case": case,
        "intent": intent,
        "identity": identity,
        "resources": resources,
        "status": status,
        "steps": entries,
    }


def predict(case: Dict[str, str]) -> Tuple[str, List[dict]]:
    """Return (deletion_status, steps) for one row of deletion_requests.csv."""
    trail = audit_case(case)
    return trail["status"], [step for step, _ in trail["steps"]]


def format_trail(trail: Dict[str, Any]) -> str:
    case, identity, intent = trail["case"], trail["identity"], trail["intent"]
    header = [
        "%s  email=%s  audit_at=%s  reported_status=%s"
        % (case["case_id"], case["email"], case["audit_at"], case.get("reported_status", "")),
        "",
        "Intent: %s (%s): %s" % ("DELETION" if intent["deletion"] else "NOT a deletion", intent["source"], intent["reason"]),
        "  request: %s" % case["request_text"],
        "",
    ]
    if identity is None:
        return "\n".join(header + ["Status: NOT_APPLICABLE (no audit performed)", "Steps: none"])
    lines = header + [
        "Identity: user_ids=%s" % (", ".join(identity.user_ids) or "(none matched)"),
        "  identifiers: %s" % ", ".join(sorted(identity.identifiers)),
        "",
        "Linked resources: %d" % len(trail["resources"]),
    ]
    for resource in trail["resources"]:
        lines.append(
            "  %s (%s)  category=%s [%s]  rule=%s  policy=%s"
            % (
                resource["resource_id"],
                resource["system_id"],
                resource.get("category"),
                resource.get("category_source", ""),
                resource.get("rule_id"),
                resource.get("policy"),
            )
        )
        shown = resource["linked_by"][:4]
        more = len(resource["linked_by"]) - len(shown)
        lines.append("    linked by: %s%s" % ("; ".join(shown), " (+%d more)" % more if more > 0 else ""))
        if "skipped" in resource:
            lines.append("    SKIPPED: %s" % resource["skipped"])
            continue
        if resource.get("note"):
            lines.append("    NOTE: %s" % resource["note"])
        for copy in resource["copies"]:
            lines.append(
                "    copy %-36s %-10s assessed=%-20s via %s %s @ %s%s"
                % (
                    copy["copy"],
                    copy["state"],
                    copy["assessed"],
                    copy["source"],
                    copy["evidence_id"],
                    copy["observed_at"],
                    "  -> VIOLATES" if copy["violates"] else "",
                )
            )
        decided = resource["decided"]
        lines.append(
            "    decision: %s (copy %s, %s)"
            % ("VIOLATION" if resource["violates"] else "compliant", decided["copy"], decided["evidence_id"])
        )
    lines += ["", "Status: %s" % trail["status"], "Steps:"]
    for step, reason in trail["steps"]:
        lines.append(
            "  %2d %-12s %-26s %-10s %s/%s  evidence=%s  [%s %s, copy=%s, state=%s, evidence=%s%s]"
            % (
                step["step_no"],
                step["phase"],
                step["operation"],
                step["state"],
                step["system_id"],
                step["resource_id"],
                step["evidence_id"] or '""',
                reason["rule_id"] or "no rule",
                reason["policy"],
                reason["copy"],
                reason["state"],
                reason["evidence"],
                ", identity mismatch" if reason["note"] else "",
            )
        )
    return "\n".join(lines)
