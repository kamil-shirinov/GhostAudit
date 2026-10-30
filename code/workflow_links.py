"""Link primary accounts that only the deletion worker's activity stream ties to a request.

Some primary accounts carry no identifier in any sighting (a bare 404, say), so identity
linking cannot reach them. The deletion worker's own DELETE_RECORD activities against
account_service can, under two constraints that made the matching trustworthy in planning:
the worker's reported_result must agree with the request's reported_status, and each
activity serves at most one request. Pairs are assigned greedily, nearest to the report
first. Accounts that carry some user's identifier belong to that user and are never
candidates. Only request fields are used, never gold answers.

The pool is every deletion request in both request files, because the worker served all
of them and they compete for the same activities. agent.py supplies the pool, already
filtered by intent, and decides which requests need an account.
"""
import csv
from datetime import datetime
from typing import Callable, Dict, FrozenSet, Iterable, List, NamedTuple, Set, Tuple

import evidence
from identity import identifiers_by_user

ACTIVITY_PATH = evidence.DATASET_DIR / "system_activity.csv"
POOL_PATHS = (evidence.DATASET_DIR / "deletion_requests.csv", evidence.DATASET_DIR / "sample_deletion_requests.csv")

# reported_status -> worker results that agree with it.
COMPATIBLE_RESULTS = {
    "COMPLETE": ("SUCCESS",),
    "NOT_STARTED": ("FAILED",),
    "PARTIALLY_COMPLETE": ("PARTIAL", "SUCCESS"),
}

Case = Dict[str, str]


class WorkflowLink(NamedTuple):
    case_id: str
    resource_id: str
    activity_id: str
    reported_result: str
    minutes_before_report: float

    def describe(self) -> str:
        return "linked via %s (deletion worker, %s, %d min before the report)" % (
            self.activity_id,
            self.reported_result,
            round(self.minutes_before_report),
        )


class Assignment(NamedTuple):
    needed: FrozenSet[str]  # case_ids of pool requests with no identifier-linked account
    links: Dict[str, WorkflowLink]  # case_id -> assigned account


def _time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def read_pool() -> List[Case]:
    cases = []
    for path in POOL_PATHS:
        with open(str(path), newline="", encoding="utf-8-sig") as handle:
            cases.extend(csv.DictReader(handle))
    return cases


def owned_accounts(sightings: Iterable[Tuple[str, FrozenSet[str]]]) -> Set[str]:
    """Account resource_ids with a sighting that carries some user's identifier.
    sightings: (resource_id, identifiers) pairs."""
    owner = {}  # type: Dict[str, str]
    for user_id, identifiers in sorted(identifiers_by_user().items()):
        for identifier in identifiers:
            owner.setdefault(identifier, user_id)
    return {
        resource_id
        for resource_id, identifiers in sightings
        if evidence.system_for(resource_id) == "account_service" and any(i in owner for i in identifiers)
    }


def _worker_deletes() -> List[Dict[str, str]]:
    with open(str(ACTIVITY_PATH), newline="", encoding="utf-8-sig") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["source_system"] == "deletion_worker"
            and row["event_type"] == "DELETE_RECORD"
            and row["target_system"] == "account_service"
        ]


def assign(pool: List[Case], needs_account: Callable[[Case], bool], owned: Set[str]) -> Assignment:
    """Greedy one-to-one assignment of worker activities to the pool requests that need one."""
    needed = [case for case in pool if needs_account(case)]
    activities = [row for row in _worker_deletes() if row["resource_id"] not in owned]

    pairs = []  # (gap to the report, case_id, activity_id, activity row)
    for case in needed:
        compatible = COMPATIBLE_RESULTS.get(case["reported_status"], ())
        requested, reported = _time(case["requested_at"]), _time(case["reported_at"])
        for row in activities:
            event = _time(row["event_time"])
            if requested <= event <= reported and row["reported_result"] in compatible:
                pairs.append((reported - event, case["case_id"], row["activity_id"], row))
    pairs.sort(key=lambda pair: pair[:3])

    links = {}  # type: Dict[str, WorkflowLink]
    used = set()  # type: Set[str]
    for gap, case_id, activity_id, row in pairs:
        if case_id in links or activity_id in used:
            continue
        used.add(activity_id)
        links[case_id] = WorkflowLink(case_id, row["resource_id"], activity_id, row["reported_result"], gap.total_seconds() / 60.0)
    return Assignment(frozenset(case["case_id"] for case in needed), links)
