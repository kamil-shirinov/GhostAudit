"""The only module that calls the Anthropic API.

complete_json() and complete_json_image() (one screenshot plus text) are the entry points.
Every call goes through here so that
- a repeated prompt is answered from code/cache/llm_cache.json and makes no API call,
- every real call is appended to code/cache/llm_usage.jsonl (the source for usage_report.md),
  tagged with the run_id and entrypoint of the script that made it (start_run),
- a finished full run is recorded in code/cache/llm_runs.jsonl, even if it made no call (record_run),
- new calls are refused once the ledger total reaches the spend cap.

With no API key, an API failure or a spend-cap hit it raises LLMUnavailable, so callers
can fall back. The API key is read from the environment or .env and is never printed or
logged.
"""
import base64
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from dotenv import load_dotenv

MODEL = "claude-haiku-5-5"

# USD per million tokens, for prompts up to 100K tokens.
PRICES_SOURCE = "platform.claude.com/docs/en/docs/about-claude/pricing"
PRICES = {MODEL: {"input": 0.10, "output": 0.50}}

DEFAULT_SPEND_CAP_USD = 1.00  # override with LLM_SPEND_CAP_USD
REQUEST_TIMEOUT_SECONDS = 120.0
JSON_INSTRUCTION = "\n\nRespond with a single JSON object and nothing else: no prose and no code fences."

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
CACHE_DIR = CODE_DIR / "cache"
CACHE_PATH = CACHE_DIR / "llm_cache.json"
LEDGER_PATH = CACHE_DIR / "llm_usage.jsonl"
RUNS_PATH = CACHE_DIR / "llm_runs.jsonl"

IMAGE_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

_FENCE = re.compile(r"^```[A-Za-z]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)

Content = Union[str, List[Dict[str, Any]]]

_cache = None  # type: Optional[Dict[str, Any]]
_client = None  # type: Any
_effort_supported = True
_run = None  # type: Optional[Dict[str, Any]]


class LLMUnavailable(RuntimeError):
    """No key, an API failure, a spend-cap hit, or unusable model output."""


def _load_env() -> None:
    load_dotenv(str(REPO_ROOT / ".env"))  # never overrides variables already set


def spend_cap_usd() -> float:
    _load_env()
    raw = os.environ.get("LLM_SPEND_CAP_USD")
    return float(raw) if raw else DEFAULT_SPEND_CAP_USD


def ledger_total_usd() -> float:
    if not LEDGER_PATH.exists():
        return 0.0
    total = 0.0
    with open(str(LEDGER_PATH), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                total += json.loads(line)["cost_usd"]
    return total


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    price = PRICES[MODEL]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1000000.0


def _cache_key(system: str, user: str, image_sha256: Optional[str] = None) -> str:
    parts = [MODEL, system, user] + ([image_sha256] if image_sha256 else [])  # text keys unchanged
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()


def _load_cache() -> Dict[str, Any]:
    global _cache
    if _cache is None:
        if CACHE_PATH.exists():
            with open(str(CACHE_PATH), encoding="utf-8") as handle:
                _cache = json.load(handle)
        else:
            _cache = {}
    return _cache


def _store(key: str, value: Dict[str, Any]) -> None:
    cache = _load_cache()
    cache[key] = value
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_suffix(".tmp")
    with open(str(temporary), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(CACHE_PATH))


def _get_client() -> Any:
    global _client
    if _client is None:
        _load_env()
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set (put it in .env or the environment)")
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailable("the anthropic package is not installed") from exc
        _client = anthropic.Anthropic(api_key=key, timeout=REQUEST_TIMEOUT_SECONDS)
    return _client


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def start_run(entrypoint: str) -> str:
    """Tag every call from now on with a new run id (the UTC start time) and this entrypoint.

    Each script that can reach the API calls this once, at start: main.py with "main" (or
    "explain" for --explain), the evaluation scripts with "evaluation". Ledger entries from
    before this existed have no run_id and count as development. A call made with no
    start_run() gets a run of its own with entrypoint "unspecified", so it is never untagged.
    """
    global _run
    _run = _new_run(entrypoint)
    return _run["run_id"]


def _new_run(entrypoint: str) -> Dict[str, Any]:
    now = _utc_now()
    return {"run_id": now, "entrypoint": entrypoint, "started_at": now, "cache_hits": 0}


def _current_run() -> Dict[str, Any]:
    global _run
    if _run is None:
        _run = _new_run("unspecified")
    return _run


def record_run(requests: int) -> None:
    """Append the current run to code/cache/llm_runs.jsonl once it has finished its work.

    The ledger only has a line per call, so a run served entirely from the cache would leave
    no trace in it; this is how the usage report finds the final run. cache_hits counts the
    answers read from the cache instead of the API.
    """
    run = _current_run()
    entry = {
        "run_id": run["run_id"],
        "entrypoint": run["entrypoint"],
        "started_at": run["started_at"],
        "finished_at": _utc_now(),
        "requests": requests,
        "cache_hits": run["cache_hits"],
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(RUNS_PATH), "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _record_usage(purpose: str, input_tokens: int, output_tokens: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    run = _current_run()
    entry = {
        "timestamp": _utc_now(),
        "run_id": run["run_id"],
        "entrypoint": run["entrypoint"],
        "purpose": purpose,
        "model": MODEL,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(estimate_cost_usd(input_tokens, output_tokens), 8),
    }
    with open(str(LEDGER_PATH), "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _call(system: str, content: Content, purpose: str, max_tokens: int) -> str:
    """One real API call with the given user content: returns the text, and records the usage."""
    global _effort_supported
    client = _get_client()
    spent, cap = ledger_total_usd(), spend_cap_usd()
    if spent >= cap:
        raise LLMUnavailable("spend cap reached: $%.4f spent of $%.2f" % (spent, cap))

    import anthropic

    # No temperature, top_p or top_k: claude-haiku-5-5 rejects them with a 400 ("temperature: Extra inputs
    # are not permitted", tested 2026-10-24 with and without effort), and anthropic 2.3.0's create() has
    # no such argument. Repeatability comes from the response cache, not from sampling settings.
    kwargs = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": content}],
    }  # type: Dict[str, Any]
    try:
        try:
            if _effort_supported:
                response = client.messages.create(output_config={"effort": "low"}, **kwargs)
            else:
                response = client.messages.create(**kwargs)
        except anthropic.BadRequestError as exc:
            if not (_effort_supported and "effort" in str(exc).lower()):
                raise
            _effort_supported = False  # this model rejects the effort setting
            response = client.messages.create(**kwargs)
    except anthropic.APIError as exc:
        message = str(exc).replace(client.api_key, "[REDACTED]")[:300]
        raise LLMUnavailable("Anthropic API error (%s): %s" % (type(exc).__name__, message)) from exc

    _record_usage(purpose, response.usage.input_tokens, response.usage.output_tokens)
    return "".join(block.text for block in response.content if block.type == "text")


def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object, got %s" % type(value).__name__)
    return value


def complete_json(
    system: str,
    user: str,
    purpose: str,
    max_tokens: int = 4000,
    validate: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Ask the model for a JSON object. validate(value) may raise ValueError to reject it.

    Invalid JSON, or a rejected value, is retried once; only an accepted value is cached.
    """
    return _complete_json(_cache_key(system, user), system, lambda text: text, user, purpose, max_tokens, validate)


def complete_json_image(
    system: str,
    user: str,
    image_path,
    purpose: str,
    max_tokens: int = 4000,
    validate: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """complete_json() with one image (sent as a base64 block before the text).

    The cache key includes the image's sha256, so an edited image is read again.
    """
    path = Path(image_path)
    media_type = IMAGE_MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise ValueError("%s: unsupported image type %r" % (path.name, path.suffix))
    data = path.read_bytes()
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": base64.b64encode(data).decode("ascii")},
    }
    key = _cache_key(system, user, hashlib.sha256(data).hexdigest())
    return _complete_json(
        key, system, lambda text: [image, {"type": "text", "text": text}], user, purpose, max_tokens, validate
    )


def _complete_json(
    key: str,
    system: str,
    make_content: Callable[[str], Content],
    user: str,
    purpose: str,
    max_tokens: int,
    validate: Optional[Callable[[Dict[str, Any]], None]],
) -> Dict[str, Any]:
    cached = _load_cache().get(key)
    if cached is not None:
        try:
            if validate is not None:
                validate(cached)
            _current_run()["cache_hits"] += 1
            return cached
        except ValueError:
            pass  # stale entry from an older schema: ask again

    last_error = None  # type: Optional[Exception]
    for attempt in (1, 2):
        prompt = user if attempt == 1 else user + "\n\nYour previous reply was invalid. Reply with only the JSON object."
        text = _call(system + JSON_INSTRUCTION, make_content(prompt), purpose, max_tokens)
        try:
            value = _parse_json(text)
            if validate is not None:
                validate(value)
        except ValueError as exc:
            last_error = exc
            continue
        _store(key, value)
        return value
    raise LLMUnavailable("the model did not return usable JSON after a retry: %s" % last_error)
