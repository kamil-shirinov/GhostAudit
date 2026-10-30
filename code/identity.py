"""Resolve a request email to the subject's user_ids and every identifier they have used.

A request email matches a user if it equals, case-insensitively, the user's current email
(user_identities.csv) or any old or new email value in user_audit_log.csv. The identifier
set holds those user_ids plus every current and historical email, phone and email_hash
for them, normalised exactly as evidence.py normalises identifiers in evidence.
"""
import csv
from functools import lru_cache
from typing import Dict, FrozenSet, List, NamedTuple, Set, Tuple

import evidence

IDENTITIES_PATH = evidence.DATASET_DIR / "user_identities.csv"
AUDIT_LOG_PATH = evidence.DATASET_DIR / "user_audit_log.csv"
IDENTITY_FIELDS = ("email", "phone", "email_hash")


class Identity(NamedTuple):
    email: str
    user_ids: Tuple[str, ...]
    identifiers: FrozenSet[str]


def _read(path) -> List[Dict[str, str]]:
    with open(str(path), newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=None)
def _tables() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    return _read(IDENTITIES_PATH), _read(AUDIT_LOG_PATH)


def _normalise(value: str) -> List[str]:
    return evidence.extract_identifiers([value]) if value else []


@lru_cache(maxsize=None)
def identifiers_by_user() -> Dict[str, FrozenSet[str]]:
    """user_id -> the user_id plus every current and historical email, phone and email_hash."""
    identities, audit_log = _tables()
    found = {}  # type: Dict[str, Set[str]]
    for row in identities:
        identifiers = found.setdefault(row["user_id"], set(_normalise(row["user_id"])))
        for field in IDENTITY_FIELDS:
            identifiers.update(_normalise(row[field]))
    for row in audit_log:
        identifiers = found.setdefault(row["user_id"], set(_normalise(row["user_id"])))
        if row["field_name"] in IDENTITY_FIELDS:
            identifiers.update(_normalise(row["old_value"]))
            identifiers.update(_normalise(row["new_value"]))
    return {user_id: frozenset(identifiers) for user_id, identifiers in found.items()}


def resolve_identity(email: str) -> Identity:
    target = email.strip().lower()
    identities, audit_log = _tables()

    user_ids = {row["user_id"] for row in identities if row["email"].strip().lower() == target}
    user_ids.update(
        row["user_id"]
        for row in audit_log
        if row["field_name"] == "email"
        and target in (row["old_value"].strip().lower(), row["new_value"].strip().lower())
    )

    identifiers = set(_normalise(email))  # type: Set[str]
    by_user = identifiers_by_user()
    for user_id in user_ids:
        identifiers.update(by_user[user_id])
    return Identity(target, tuple(sorted(user_ids)), frozenset(identifiers))
