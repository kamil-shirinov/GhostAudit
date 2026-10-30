"""Normalise every evidence file into one view: sightings and coverage.

Schema-agnostic by design: any record that carries a resource_id becomes a "sighting",
whatever the file format, instead of one parser per file family. That is less code and
survives formats we have not seen.

load_sightings() -> one dict per observation of a resource in a file.
load_coverage()  -> every place where a file lists the complete contents of a system at
                    a moment, so a resource missing from it is proven ABSENT at that time.

PNG screenshots are read by vision.py into records. A record (or a PDF) that prints no
resource ID is a "document"; it becomes a sighting only through a catalog alias: the data
catalog's LINK_RESOURCE_IDENTITY activities publish "<resource_id> (<alias>)", e.g.
"object_19 (delivery_confirmation_19.pdf)", and a document showing that alias is a
sighting of that resource.

Reads dataset/evidence_files.csv, dataset/evidence/ and, for aliases only,
dataset/system_activity.csv. Results are cached per dataset directory; treat the
returned dicts as read-only. unreadable_screenshots() lists PNGs that could not be read
(no cached reading and no model).

Evidence files are untrusted input, so two guards raise UnsafeEvidenceError instead of
skipping the file: a manifest path must resolve inside dataset/evidence/, and a SQL dump
may not ATTACH or DETACH a database (that would let it read or create files on disk).
"""
import csv
import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pypdf

import vision
from llm import LLMUnavailable

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

# resource_id prefix -> system_id. Unknown prefixes get system_id None and are reported,
# never guessed.
SYSTEM_BY_PREFIX = {
    "account": "account_service",
    "search": "search_service",
    "analytics": "analytics_platform",
    "support": "support_platform",
    "export": "export_service",
    "backup": "backup_service",
    "data_flow": "integration_service",
    "audit_record": "security_service",
    "tombstone": "identity_service",
    "cache": "cache_service",
    "email_delivery": "email_delivery_service",
    "legal_hold": "legal_service",
    "marketing": "marketing_platform",
    "object": "object_storage",
}

# Fallback system for an empty SQL table, by table-name prefix (customer_profiles* etc.).
SYSTEM_BY_TABLE_PREFIX = (
    ("customer_profiles", "account_service"),
    ("customer_search_documents", "search_service"),
    ("search_documents", "search_service"),
    ("customer_analytics_profiles", "analytics_platform"),
    ("support_records", "support_platform"),
    ("customer_exports", "export_service"),
    ("connector_subscriptions", "integration_service"),
    ("backup_restore_entries", "backup_service"),
    ("security_audit_records", "security_service"),
)

SCREENSHOT_SUFFIXES = (".png",)

_RESOURCE_ID = re.compile(
    r"\b(?:%s)_\d+\b" % "|".join(re.escape(p) for p in sorted(SYSTEM_BY_PREFIX, key=len, reverse=True))
)
_RESOURCE_SHAPE = re.compile(r"(.+)_\d+")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_USER_REF = re.compile(r"\buser_\d+\b", re.IGNORECASE)
_PHONE = re.compile(r"\+\d[\d ().-]{5,}\d")
_HASH = re.compile(r"\b[0-9a-fA-F]{64}\b")

# Own observation time of a record; ancestors may also contribute a snapshot captured_at.
_OWN_TIME_KEYS = ("observed_at", "recorded_at")
_INHERITED_TIME_KEYS = ("observed_at", "recorded_at", "captured_at")
_CONTEXT_KEYS = ("data_store", "system", "collection")

_RESULT_STATES = {
    "FOUND": "PRESENT",
    "NOT_FOUND": "ABSENT",
    "HIT": "PRESENT",
    "MISS": "ABSENT",
    "IDENTIFIABLE": "IDENTIFIABLE",
    "ANONYMIZED": "ANONYMIZED",
    "REDACTED": "REDACTED",
    "ACTIVE": "ACTIVE",
    "DISABLED": "DISABLED",
    "BLOCKED": "BLOCKED",
}

Sighting = Dict[str, Any]
Coverage = Dict[str, Any]
Document = Dict[str, Any]  # evidence that names no resource: see _resolve_aliases()


class UnsafeEvidenceError(ValueError):
    """An evidence file or manifest entry tried to reach outside the evidence it describes."""


def resource_prefix(resource_id: str) -> str:
    match = _RESOURCE_SHAPE.fullmatch(resource_id)
    return match.group(1) if match else resource_id


def system_for(resource_id: str) -> Optional[str]:
    return SYSTEM_BY_PREFIX.get(resource_prefix(resource_id))


def extract_identifiers(texts) -> List[str]:
    """Sorted, normalised emails, user_<n> references, phone numbers and 64-hex hashes."""
    found = set()
    for text in texts:
        found.update(match.lower() for match in _EMAIL.findall(text))
        found.update(match.lower() for match in _USER_REF.findall(text))
        found.update(match.lower() for match in _HASH.findall(text))
        for match in _PHONE.findall(text):
            digits = re.sub(r"\D", "", match)
            if len(digits) >= 7:
                found.add("+" + digits)
    return sorted(found)


def flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    """Nested objects and lists become dotted keys (profile.email, recent_activity.0.event)."""
    out = {}  # type: Dict[str, Any]
    _flatten_into(value, prefix, out)
    return out


def _flatten_into(value: Any, prefix: str, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        if not value and prefix:
            out[prefix] = {}
        for key, child in value.items():
            _flatten_into(child, "%s.%s" % (prefix, key) if prefix else key, out)
    elif isinstance(value, list):
        if not value and prefix:
            out[prefix] = []
        for index, child in enumerate(value):
            _flatten_into(child, "%s.%d" % (prefix, index) if prefix else str(index), out)
    else:
        out[prefix] = value


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def _as_flag(value: Any) -> Optional[bool]:
    text = _as_text(value)
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    return None


def infer_state(fields: Dict[str, Any]) -> str:
    """Best-effort state of the resource from its fields; see the rule order below."""
    result = fields.get("result")
    if isinstance(result, str) and result.strip().upper() in _RESULT_STATES:
        return _RESULT_STATES[result.strip().upper()]
    for key in ("http_status", "status_code"):
        if _as_text(fields.get(key)) == "404":
            return "ABSENT"
    for key in ("rows", "hits", "total_hits", "hits.total"):
        if _as_text(fields.get(key)) == "0":
            return "ABSENT"
    enabled = _as_flag(fields.get("enabled"))
    if enabled is not None:
        return "ACTIVE" if enabled else "DISABLED"
    restore_allowed = _as_flag(fields.get("restore_allowed"))
    if restore_allowed is not None:
        return "ACTIVE" if restore_allowed else "BLOCKED"
    if _as_flag(fields.get("eligible")) is True:
        return "ACTIVE"
    if _as_flag(fields.get("deletion_block")) is True:
        return "BLOCKED"
    for key in ("available", "download_available", "archive_present", "present"):
        if _as_flag(fields.get(key)) is False:
            return "ABSENT"
    return "PRESENT"


def _first_text(mapping: Dict[str, Any], keys) -> Optional[str]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _sighting(
    entry: Dict[str, str],
    resource_id: str,
    fields: Dict[str, Any],
    observed_at: Optional[str],
    context: Dict[str, str],
) -> Sighting:
    texts = [value for value in fields.values() if isinstance(value, str)]
    linked = {match for text in texts for match in _RESOURCE_ID.findall(text)}
    linked.discard(resource_id)
    return {
        "evidence_id": entry["file_id"],
        "captured_at": entry["captured_at"],
        "observed_at": observed_at or entry["captured_at"],
        "resource_id": resource_id,
        "system_id": system_for(resource_id),
        "context": context,
        "fields": fields,
        "identifiers": extract_identifiers(texts),
        "linked_resource_ids": sorted(linked),
        "state": infer_state(fields),
    }


# ---------------------------------------------------------------- JSON

def _json_evidence(entry: Dict[str, str], path: Path) -> Tuple[List[Sighting], List[Coverage]]:
    with open(str(path), encoding="utf-8") as handle:
        data = json.load(handle)
    sightings = []  # type: List[Sighting]

    def walk(node: Any, keys: List[str], inherited_time: Optional[str], inherited_context: Dict[str, str]) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item, keys, inherited_time, inherited_context)
            return
        if not isinstance(node, dict):
            return
        context = dict(inherited_context)
        for key in _CONTEXT_KEYS:
            if isinstance(node.get(key), str):
                context[key] = node[key]
        if isinstance(node.get("resource_id"), str):
            record_context = dict(context, path=".".join(keys))
            observed_at = _first_text(node, _OWN_TIME_KEYS) or inherited_time
            sightings.append(_sighting(entry, node["resource_id"], flatten(node), observed_at, record_context))
        child_time = _first_text(node, _INHERITED_TIME_KEYS) or inherited_time
        for key, child in node.items():
            if isinstance(child, (dict, list)):
                walk(child, keys + [key], child_time, context)

    walk(data, [], None, {})

    # Snapshot blocks list a data store's complete contents at captured_at, even when empty.
    coverage = []  # type: List[Coverage]
    blocks = data.get("snapshots") if isinstance(data, dict) else None
    for block in blocks if isinstance(blocks, list) else []:
        if not (isinstance(block, dict) and isinstance(block.get("data_store"), str)):
            continue
        records = block.get("records") if isinstance(block.get("records"), list) else []
        coverage.append({
            "evidence_id": entry["file_id"],
            "system_id": block["data_store"],
            "observed_at": block.get("captured_at") or entry["captured_at"],
            "resource_ids": sorted(
                record["resource_id"]
                for record in records
                if isinstance(record, dict) and isinstance(record.get("resource_id"), str)
            ),
            "context": {"collection": block.get("collection")},
        })
    return sightings, coverage


# ---------------------------------------------------------------- SQL

_SQL_CAPTURED_AT = re.compile(r"^--\s*Captured at:\s*(\S+)", re.MULTILINE)


def _sql_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Row columns, with any JSON-object column (record_json) merged in, flattened."""
    fields = dict(row)
    for column, value in row.items():
        if not (column.endswith("_json") and isinstance(value, str)):
            continue
        try:
            parsed = json.loads(value)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            del fields[column]
            fields.update(flatten(parsed))
    return fields


def _table_system(table: str) -> Optional[str]:
    for prefix, system in SYSTEM_BY_TABLE_PREFIX:
        if table.startswith(prefix):
            return system
    return None


def _sql_evidence(entry: Dict[str, str], path: Path) -> Tuple[List[Sighting], List[Coverage]]:
    script = path.read_text(encoding="utf-8")
    header = _SQL_CAPTURED_AT.search(script)
    file_snapshot_time = header.group(1) if header else None

    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    refused = []  # type: List[int]

    def authorizer(action: int, *_: Any) -> int:
        # SQLite also reports VACUUM INTO (which writes a file) as an ATTACH.
        if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
            refused.append(action)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    db.set_authorizer(authorizer)
    try:
        try:
            db.executescript(script)
        except sqlite3.DatabaseError as exc:
            if refused:
                raise UnsafeEvidenceError(
                    "%s: the SQL dump tries to ATTACH or DETACH a database; refusing to load it" % entry["file_id"]
                ) from exc
            raise
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY rowid")]
        # snapshot_scope names the systems this file captures completely, with their capture time.
        scope = {}  # type: Dict[str, str]
        if "snapshot_scope" in tables:
            scope = {row["origin_system"]: row["captured_at"] for row in db.execute("SELECT * FROM snapshot_scope")}

        sightings = []  # type: List[Sighting]
        coverage = []  # type: List[Coverage]
        for table in tables:
            quoted = '"%s"' % table.replace('"', '""')
            columns = [row[1] for row in db.execute("PRAGMA table_info(%s)" % quoted)]
            if "resource_id" not in columns:
                continue
            listed = {}  # type: Dict[str, List[str]]
            row_count = 0
            for row in db.execute("SELECT * FROM %s" % quoted):
                row_count += 1
                resource_id = row["resource_id"]
                if not isinstance(resource_id, str):
                    continue
                system = system_for(resource_id)
                fields = _sql_fields(dict(row))
                context = {"table": table}
                if isinstance(fields.get("origin_system"), str):
                    context["system"] = fields["origin_system"]
                if isinstance(fields.get("collection_name"), str):
                    context["collection"] = fields["collection_name"]
                observed_at = _first_text(fields, _OWN_TIME_KEYS) or scope.get(system or "") or file_snapshot_time
                sightings.append(_sighting(entry, resource_id, fields, observed_at, context))
                if system is not None:
                    listed.setdefault(system, []).append(resource_id)

            if row_count == 0:
                fallback = _table_system(table)
                if fallback is not None:
                    listed[fallback] = []
            for system, resource_ids in listed.items():
                if scope and system not in scope:
                    continue  # not captured completely by this file, so absence proves nothing
                coverage.append({
                    "evidence_id": entry["file_id"],
                    "system_id": system,
                    "observed_at": scope.get(system) or file_snapshot_time or entry["captured_at"],
                    "resource_ids": sorted(resource_ids),
                    "context": {"table": table},
                })
        return sightings, coverage
    finally:
        db.close()


# ---------------------------------------------------------------- TXT handover

_HANDOVER_FIELDS = ("recorded_at", "system", "resource_id", "record_type", "result", "subject_reference")


def _txt_evidence(entry: Dict[str, str], path: Path) -> List[Sighting]:
    """Lines of the form '- time | system | resource_id | record_type | result | subject_reference'."""
    sightings = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        parts = [part.strip() for part in line[2:].split("|")]
        if len(parts) != len(_HANDOVER_FIELDS):
            continue
        fields = dict(zip(_HANDOVER_FIELDS, parts))
        sightings.append(
            _sighting(entry, fields["resource_id"], fields, fields["recorded_at"], {"system": fields["system"]})
        )
    return sightings


# ---------------------------------------------------------------- logs

_LOG_LINE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2}))\s+(.*)")
_KEY_VALUE = re.compile(r'([A-Za-z_][\w.-]*)=("(?:[^"\\]|\\.)*"|\S+)')
# Positional headers in front of the key=value pairs.
_BRACKET_HEADER = re.compile(
    r"\[(?P<level>[^\]]+)\]\s+\[(?P<service>[^\]]+)\]\s+\[(?P<trace_id>[^\]]+)\]\s+(?P<event>[^\s:]+):?\s*(?P<rest>.*)"
)
_PIPE_HEADER = re.compile(
    r"(?P<level>[^|]+?)\s+\|\s+(?P<service>[^|]+?)\s+\|\s+(?P<trace_id>[^|]+?)\s+\|\s+(?P<event>[^|]+?)\s+\|\s*(?P<rest>.*)"
)
_SYSLOG_HEADER = re.compile(r"<\d+>\d*\s+(?P<host>\S+)\s+(?P<service>\S+)\s+(?P<trace_id>\S+)\s+\S+\s+\S+\s+(?P<rest>.*)")
_LEVELS = {"TRACE", "DEBUG", "INFO", "NOTICE", "WARN", "WARNING", "ERROR", "CRITICAL", "FATAL"}


def _unquote(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        return re.sub(r"\\(.)", r"\1", raw[1:-1])
    return raw


def _parse_pairs(text: str) -> Dict[str, Any]:
    fields = {key: _unquote(raw) for key, raw in _KEY_VALUE.findall(text)}  # type: Dict[str, Any]
    # cloud_runtime_export nests the real pairs inside message="..." with escaped quotes.
    message = fields.get("message")
    if isinstance(message, str) and _KEY_VALUE.search(message):
        del fields["message"]
        for key, value in _parse_pairs(message).items():
            fields.setdefault(key, value)
    return fields


def _parse_log_payload(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            return flatten(parsed)
    for header in (_BRACKET_HEADER, _PIPE_HEADER, _SYSLOG_HEADER):
        match = header.fullmatch(text)
        if match:
            positional = match.groupdict()
            fields = _parse_pairs(positional.pop("rest"))
            for key, value in positional.items():
                if value and value.strip() != "-":
                    fields.setdefault(key, value.strip())
            return fields
    fields = _parse_pairs(text)
    first_word = text.split(None, 1)[0] if text else ""
    if first_word.upper() in _LEVELS:
        fields.setdefault("level", first_word)
    return fields


def _log_evidence(entry: Dict[str, str], path: Path) -> List[Sighting]:
    """One record per trace: 'started' lines carry the lookup key, 'completed' lines the
    resource_id and result. A line with a resource_id but no trace id is its own record."""
    units = []  # type: List[List[Tuple[str, Dict[str, Any]]]]
    unit_by_trace = {}  # type: Dict[str, int]
    with open(str(path), encoding="utf-8") as handle:
        for line in handle:
            match = _LOG_LINE.fullmatch(line.strip())
            if not match:
                continue
            timestamp, payload = match.groups()
            fields = _parse_log_payload(payload)
            trace = fields.get("trace_id") or fields.get("requestId")
            if isinstance(trace, str) and trace:
                if trace not in unit_by_trace:
                    unit_by_trace[trace] = len(units)
                    units.append([])
                units[unit_by_trace[trace]].append((timestamp, fields))
            elif isinstance(fields.get("resource_id"), str):
                units.append([(timestamp, fields)])

    sightings = []
    for lines in units:
        resource_ids = []  # type: List[str]
        for _, fields in lines:
            resource_id = fields.get("resource_id")
            if isinstance(resource_id, str) and resource_id and resource_id not in resource_ids:
                resource_ids.append(resource_id)
        shared = [fields for _, fields in lines if not fields.get("resource_id")]
        for resource_id in resource_ids:
            own = [(timestamp, fields) for timestamp, fields in lines if fields.get("resource_id") == resource_id]
            merged = {}  # type: Dict[str, Any]
            # Only a line carrying the resource_id may set result, so a cache_lookup "miss"
            # elsewhere in the trace cannot leak into the record.
            for fields in shared:
                merged.update((key, value) for key, value in fields.items() if key != "result")
            for _, fields in own:
                merged.update(fields)
            last_timestamp, last_fields = own[-1]
            context = {
                "service": _first_text(merged, ("service", "function", "logger")),
                "event": _first_text(last_fields, ("event",)),
            }
            sightings.append(
                _sighting(entry, resource_id, merged, last_timestamp, {k: v for k, v in context.items() if v})
            )
    return sightings


# ---------------------------------------------------------------- PDF

def _document(entry: Dict[str, str], fields: Dict[str, Any], context: Dict[str, str], texts, own_keys) -> Document:
    """texts: where a non-user alias may appear. own_keys: the item's own keys (lowercase),
    the only place a user_<n> alias may match."""
    return {
        "entry": entry,
        "fields": fields,
        "context": context,
        "texts": [text for text in texts if isinstance(text, str)],
        "own_keys": frozenset(key.lower() for key in own_keys),
    }


def _pdf_evidence(entry: Dict[str, str], path: Path) -> Tuple[List[Sighting], List[Document]]:
    """One record per resource ID in the text, each carrying the whole document's text.
    A PDF with no resource ID is one document; its text is free text, so it has no own keys."""
    reader = pypdf.PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    resource_ids = []  # type: List[str]
    for resource_id in _RESOURCE_ID.findall(text):
        if resource_id not in resource_ids:
            resource_ids.append(resource_id)
    context = {"document": path.name}
    if not resource_ids:
        return [], [_document(entry, {"text": text}, context, [text], ())]
    return [_sighting(entry, resource_id, {"text": text}, None, context) for resource_id in resource_ids], []


# ---------------------------------------------------------------- PNG screenshots

def _screenshot_evidence(entry: Dict[str, str], path: Path) -> Tuple[List[Sighting], List[Document]]:
    """One record per item vision.py read off the screen, observed at the capture time.

    A printed ID that is not resource-shaped is kept as an alias, never trusted as an ID; if
    exactly one printed alias is resource-shaped, that alias is the item's ID.
    """
    screenshot = vision.extract_screenshot(path)
    context = {"source": "screenshot"}
    if screenshot["screen"]:
        context["screen"] = screenshot["screen"]
    title_keys = _USER_REF.findall(screenshot["screen"] or "") if len(screenshot["records"]) == 1 else []

    sightings = []  # type: List[Sighting]
    documents = []  # type: List[Document]
    for record in screenshot["records"]:
        resource_id, aliases = record["resource_id"], list(record["aliases"])
        if resource_id is not None and not _RESOURCE_ID.fullmatch(resource_id):
            aliases.insert(0, resource_id)
            resource_id = None
        shaped = [alias for alias in aliases if _RESOURCE_ID.fullmatch(alias)]
        if resource_id is None and len(shaped) == 1:
            resource_id = shaped[0]
            aliases.remove(resource_id)
        fields = dict(record["fields"])
        fields.update(flatten(aliases, "aliases"))
        if resource_id is not None:
            sightings.append(_sighting(entry, resource_id, fields, entry["captured_at"], context))
            continue
        user_reference = fields.get("user_reference")
        own_keys = aliases + title_keys + ([user_reference] if isinstance(user_reference, str) else [])
        documents.append(_document(entry, fields, context, list(fields.values()), own_keys))
    return sightings, documents


# ---------------------------------------------------------------- catalog aliases

_CATALOG_ALIAS = re.compile(r"(%s)\s+\(([^()]+)\)" % _RESOURCE_ID.pattern)


def read_catalog_aliases(dataset_dir=None) -> List[Tuple[str, str, str]]:
    """(alias, resource_id, activity_id) from every "<resource_id> (<alias>)" in a
    LINK_RESOURCE_IDENTITY activity, leaving out aliases equal to the resource_id."""
    root = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR
    aliases = []  # type: List[Tuple[str, str, str]]
    with open(str(root / "system_activity.csv"), newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["event_type"] != "LINK_RESOURCE_IDENTITY":
                continue
            for resource_id, alias in _CATALOG_ALIAS.findall(row["details"]):
                alias = alias.strip()
                if alias and alias.lower() != resource_id.lower():
                    aliases.append((alias, resource_id, row["activity_id"]))
    return aliases


def _alias_matches(alias: str, document: Document) -> bool:
    # A user_<n> reference matches only the item's own key: free text such as a register
    # listing several users would otherwise tie every one of them to the resource.
    if _USER_REF.fullmatch(alias):
        return alias.lower() in document["own_keys"]
    pattern = re.compile(r"(?<![\w.-])%s(?![\w-])" % re.escape(alias), re.IGNORECASE)
    return any(pattern.search(text) for text in document["texts"])


def _resolve_aliases(documents: List[Document], aliases: List[Tuple[str, str, str]]) -> List[Sighting]:
    """A document showing a catalog alias becomes a sighting of the aliased resource."""
    sightings = []  # type: List[Sighting]
    for document in documents:
        resolved = set()
        for alias, resource_id, activity_id in aliases:
            if resource_id in resolved or not _alias_matches(alias, document):
                continue
            resolved.add(resource_id)
            entry = document["entry"]
            context = dict(document["context"], alias=alias, alias_activity=activity_id)
            sightings.append(_sighting(entry, resource_id, document["fields"], entry["captured_at"], context))
    return sightings


# ---------------------------------------------------------------- entry points

def read_manifest(dataset_dir=None) -> List[Dict[str, str]]:
    root = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR
    with open(str(root / "evidence_files.csv"), newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def evidence_path(dataset_dir, entry: Dict[str, str]) -> Path:
    """The manifest entry's file, resolved; it must lie inside <dataset_dir>/evidence/.

    Resolving first means ../ segments, absolute paths and symlinks that lead elsewhere are
    all caught, whatever the manifest says.
    """
    root = Path(dataset_dir)
    allowed = (root / "evidence").resolve()
    path = (root / entry["file_path"]).resolve()
    if allowed not in path.parents:
        raise UnsafeEvidenceError(
            "%s: manifest path %r resolves outside %s" % (entry["file_id"], entry["file_path"], allowed)
        )
    return path


@lru_cache(maxsize=None)
def _extract(dataset_dir: str) -> Tuple[Tuple[Sighting, ...], Tuple[Coverage, ...], Tuple[Tuple[str, str], ...]]:
    sightings = []  # type: List[Sighting]
    coverage = []  # type: List[Coverage]
    documents = []  # type: List[Document]
    unreadable = []  # type: List[Tuple[str, str]]
    for entry in read_manifest(dataset_dir):
        path = evidence_path(dataset_dir, entry)
        suffix = path.suffix.lower()
        if suffix == ".json":
            found, covered = _json_evidence(entry, path)
            sightings.extend(found)
            coverage.extend(covered)
        elif suffix == ".sql":
            found, covered = _sql_evidence(entry, path)
            sightings.extend(found)
            coverage.extend(covered)
        elif suffix == ".txt":
            sightings.extend(_txt_evidence(entry, path))
        elif suffix == ".log":
            sightings.extend(_log_evidence(entry, path))
        elif suffix in (".pdf",) + SCREENSHOT_SUFFIXES:
            try:
                found, docs = (_pdf_evidence if suffix == ".pdf" else _screenshot_evidence)(entry, path)
            except LLMUnavailable as exc:
                unreadable.append((entry["file_id"], str(exc)))
                continue
            sightings.extend(found)
            documents.extend(docs)
        else:
            raise ValueError("%s: no extractor for %r files" % (entry["file_id"], suffix))
    sightings.extend(_resolve_aliases(documents, read_catalog_aliases(dataset_dir)))
    return tuple(sightings), tuple(coverage), tuple(unreadable)


def _root(dataset_dir) -> str:
    return str(Path(dataset_dir) if dataset_dir is not None else DATASET_DIR)


def load_sightings(dataset_dir=None) -> List[Sighting]:
    return list(_extract(_root(dataset_dir))[0])


def load_coverage(dataset_dir=None) -> List[Coverage]:
    return list(_extract(_root(dataset_dir))[1])


def unreadable_screenshots(dataset_dir=None) -> List[Tuple[str, str]]:
    """(file_id, reason) for each screenshot with no cached reading that the model could not read."""
    return list(_extract(_root(dataset_dir))[2])
