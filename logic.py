"""Pure Vector Store policy: settings, message parsing, chunking and deterministic identifiers.

Nothing here opens a socket, a file or the wallet. Every value is validated before
any provider call or storage write, so a refused message changes nothing.
"""

from collections.abc import Mapping
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

DEFAULTS = {
    "collection": "default", "base_url": "http://127.0.0.1:11434/v1", "model": "nomic-embed-text",
    "api_key_ref": "", "top_k": 5, "min_score": 0.0, "chunk_size": 1000, "chunk_overlap": 100,
    "allow_delete": False, "query_prefix": "", "document_prefix": "", "allow_remote": False,
    "allow_insecure_http": False, "ca_file": "", "batch_size": 32, "timeout_sec": 60,
    "max_items": 20000, "max_text_chars": 8000, "max_input_bytes": 4194304,
    "max_response_bytes": 16777216, "max_database_mb": 512}
RUNTIME_KEYS = {"execution", "runtime_path", "runtime_path_label"}
INTEGERS = (("top_k", 1, 100), ("chunk_size", 0, 100000), ("chunk_overlap", 0, 50000), ("batch_size", 1, 512),
            ("timeout_sec", 1, 600), ("max_items", 1, 200000), ("max_text_chars", 100, 100000),
            ("max_input_bytes", 1024, 33554432), ("max_response_bytes", 65536, 134217728),
            ("max_database_mb", 16, 8192))
COLLECTION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
SECRET_REF = re.compile(r"secret://workspace/[A-Za-z0-9_.-]{1,64}\Z")
SOURCE_ID = re.compile(r"[^\s#\x00-\x1f\x7f]{1,200}\Z")
ITEM_ID = re.compile(r"[^\s\x00-\x1f\x7f]{1,220}\Z")
MODEL = re.compile(r"[^\s\x00-\x1f\x7f]{1,200}\Z")
HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
MAX_DOCUMENTS = 1000
MAX_CHUNKS = 4096
MAX_DOCUMENT_CHARS = 2000000
MAX_METADATA_BYTES = 8192
MAX_FILTER_KEYS = 16
MAX_DELETE_IDS = 1000
MAX_DIMENSION = 8192
MAX_OUTPUT_BYTES = 16777216


class VectorError(ValueError):
    """A content-free diagnostic with a stable machine-readable code."""

    def __init__(self, code, message, *, retryable=False, **details):
        super().__init__(message)
        self.code, self.retryable, self.details = code, retryable, details

    def envelope(self, operation="", port=""):
        """Public error document; never contains a credential or a raw provider body."""
        result = {"code": self.code, "message": str(self), "retryable": self.retryable}
        if operation:
            result["operation"] = operation
        if port:
            result["input"] = port
        result.update(self.details)
        return result


def fail(code, message, **details):
    """Raise a refusal before any side effect."""
    raise VectorError(code, message, **details)


def thaw(value):
    """Normalize immutable preparation snapshots (managed hosts) into plain JSON values."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(item) for item in value]
    return value


def checked(value, limit, *, depth=32, label="JSON"):
    """Bound finite JSON depth, size and types; return its compact UTF-8 text."""
    pending, count = [(value, 0)], 0
    while pending:
        item, level = pending.pop()
        count += 1
        if level > depth or count > 200000:
            fail("input", f"{label} exceeds the depth or element limits.")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                fail("input", f"{label} object keys must be strings.")
            pending.extend((child, level + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, level + 1) for child in item)
        elif item is not None and type(item) not in (str, int, float, bool):
            fail("input", f"{label} must contain JSON values only.")
    try:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
        size = len(text.encode("utf-8"))
    except (ValueError, TypeError, UnicodeError, RecursionError):
        fail("input", f"{label} must be finite UTF-8 JSON.")
    if size > limit:
        fail("input", f"{label} exceeds its size limit of {limit} bytes.")
    return text


def _pairs(items):
    """Duplicate JSON keys are ambiguous; refuse instead of keeping the last one."""
    result = {}
    for key, value in items:
        if key in result:
            fail("input", "Duplicate JSON property.")
        result[key] = value
    return result


def _constant(name):
    fail("input", "NaN and Infinity are not valid JSON numbers.")


def decode(value, limit):
    """Plain text stays text; a message starting with { or [ must be valid strict JSON."""
    if not isinstance(value, str):
        checked(value, limit)
        return value, True
    if len(value) > limit or len(value.encode("utf-8")) > limit:
        fail("input", f"The message exceeds max_input_bytes ({limit} bytes).")
    if not value.lstrip().startswith(("{", "[")):
        return value, False
    try:
        parsed = json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant)
    except VectorError:
        raise
    except (ValueError, RecursionError):
        fail("input", "A message starting with { or [ must be valid JSON; send plain text or {\"text\": ...}.")
    checked(parsed, limit)
    return parsed, True


def _loopback_host(host):
    """Only literal loopback addresses and localhost are local without DNS interpretation."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def endpoint(config):
    """Validate the configured OpenAI-compatible API root and return its request target."""
    raw = config["base_url"]
    if not isinstance(raw, str) or not 1 <= len(raw) <= 2048 or "\\" in raw or any(ord(c) < 33 or ord(c) > 126 for c in raw):
        fail("config", "base_url must be an ASCII http(s) URL without spaces.")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError:
        fail("config", "base_url has an invalid host or port.")
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        fail("config", "base_url must use http or https and name a host.")
    if parsed.username is not None or parsed.password is not None:
        fail("config", "base_url must not contain credentials; use api_key_ref.")
    if parsed.query or parsed.fragment or "?" in raw or "#" in raw:
        fail("config", "base_url must not contain a query or fragment.")
    host = parsed.hostname.lower()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
        if "%" in host or host.endswith(".") or len(host) > 253 or not all(HOST_LABEL.fullmatch(label) for label in host.split(".")):
            fail("config", "base_url has an invalid host name.")
    if address is not None and (address.is_link_local or address.is_multicast or address.is_unspecified):
        fail("config", "base_url points to a link-local, multicast or unspecified address.")
    port = port if port is not None else (443 if parsed.scheme == "https" else 80)
    if not 1 <= port <= 65535:
        fail("config", "base_url has an invalid port.")
    path = parsed.path.rstrip("/")
    segments = path.split("/")[1:] if path else []
    if "%" in path or "//" in path or any(part in {"", ".", ".."} for part in segments):
        fail("config", "base_url path must be a plain API root such as /v1.")
    local = _loopback_host(host)
    if not local and not config["allow_remote"]:
        fail("config", "A non-loopback embeddings host requires allow_remote.")
    if parsed.scheme == "http" and not local:
        if not config["allow_insecure_http"]:
            fail("config", "Plain HTTP to a non-loopback host requires allow_insecure_http.")
        if config["api_key_ref"]:
            fail("config", "An API key is only sent over HTTPS or to a loopback host.")
    shown = f"[{host}]" if address is not None and address.version == 6 else host
    return {"scheme": parsed.scheme, "host": host, "port": port, "path": path + "/embeddings",
            "loopback": local, "display": f"{parsed.scheme}://{shown}:{port}{path}"}


def configuration(raw):
    """Validate every setting without touching the network, the wallet or storage."""
    if not isinstance(raw, Mapping):
        fail("config", "Settings must be an object.")
    unknown = sorted(set(raw) - set(DEFAULTS) - RUNTIME_KEYS)
    if unknown:
        fail("config", "Unknown setting: " + ", ".join(str(key) for key in unknown[:5]) + ".")
    config = {**DEFAULTS, **{key: thaw(value) for key, value in raw.items() if key in DEFAULTS}}
    if not isinstance(config["collection"], str) or not COLLECTION.fullmatch(config["collection"]):
        fail("config", "collection needs 1–64 ASCII letters, digits, dot, underscore or hyphen, starting with a letter or digit.")
    if not isinstance(config["model"], str) or not MODEL.fullmatch(config["model"]):
        fail("config", "model must be a non-empty model name without spaces (at most 200 characters).")
    for key in ("allow_delete", "allow_remote", "allow_insecure_http"):
        if type(config[key]) is not bool:
            fail("config", f"{key} must be true or false.")
    ref = config["api_key_ref"]
    if not isinstance(ref, str) or (ref and not SECRET_REF.fullmatch(ref)):
        fail("config", "api_key_ref must be empty or a wallet reference such as secret://workspace/embeddings_key; never paste the key itself.")
    for key in ("query_prefix", "document_prefix"):
        value = config[key]
        if not isinstance(value, str) or len(value) > 200 or any(ord(c) < 32 and c not in "\t" for c in value):
            fail("config", f"{key} must be at most 200 characters on one line.")
    ca = config["ca_file"]
    if not isinstance(ca, str) or len(ca) > 4096 or "\x00" in ca or (ca and not Path(ca).is_absolute()):
        fail("config", "ca_file must be empty or an absolute PEM file path.")
    for key, low, high in INTEGERS:
        if type(config[key]) is not int or not low <= config[key] <= high:
            fail("config", f"{key} must be an integer between {low} and {high}.")
    if config["chunk_size"] and config["chunk_size"] < 100:
        fail("config", "chunk_size must be 0 (no chunking) or at least 100 characters.")
    if config["chunk_size"] and config["chunk_overlap"] > config["chunk_size"] // 2:
        fail("config", "chunk_overlap must be at most half of chunk_size.")
    if config["chunk_size"] > config["max_text_chars"]:
        fail("config", "chunk_size cannot exceed max_text_chars.")
    score = config["min_score"]
    if type(score) not in (int, float) or not math.isfinite(score) or not -1 <= score <= 1:
        fail("config", "min_score must be a number between -1 and 1.")
    config["min_score"] = float(score)
    endpoint(config)
    return config


def text_value(value, limit, label="text"):
    """A non-blank string without NUL characters and within its length limit."""
    if not isinstance(value, str) or "\x00" in value or not value.strip():
        fail("input", f"{label} must be a non-empty string.")
    if len(value) > limit:
        fail("input", f"{label} exceeds {limit} characters.")
    return value


def scalar_key(value):
    """Canonical JSON for a filterable scalar: 1 equals 1.0, but true never equals 1."""
    if isinstance(value, float) and value.is_integer() and abs(value) < 2 ** 53:
        value = int(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def metadata_filter(value, *, allow_empty=True):
    """Equality filter on top-level metadata keys with scalar values only."""
    if value is None:
        value = {}
    if not isinstance(value, dict) or len(value) > MAX_FILTER_KEYS:
        fail("input", f"filter must be an object with at most {MAX_FILTER_KEYS} keys.")
    if not value and not allow_empty:
        fail("input", "A delete filter must name at least one metadata key.")
    for key, item in value.items():
        if not isinstance(key, str) or not 1 <= len(key) <= 128:
            fail("input", "Filter keys must be 1–128 character strings.")
        if item is not None and type(item) not in (str, int, float, bool):
            fail("input", "Filter values must be strings, numbers, booleans or null; nested values are not filterable.")
        if isinstance(item, float) and not math.isfinite(item):
            fail("input", "Filter numbers must be finite.")
        if isinstance(item, str) and len(item) > 1024:
            fail("input", "Filter strings are limited to 1024 characters.")
    return dict(value)


def chunk_spans(text, size, overlap):
    """Deterministic character windows, cut at whitespace when possible, with overlap."""
    total = len(text)
    if size <= 0 or total <= size:
        spans = [(0, total)]
    else:
        spans, start = [], 0
        while start < total:
            end = min(start + size, total)
            if end < total:
                floor = start + (size * 3) // 4
                cut = max(text.rfind(char, floor, end) for char in (" ", "\n", "\t", "\r"))
                if cut >= floor:
                    end = cut + 1
            spans.append((start, end))
            if end >= total:
                break
            following = max(end - overlap, start + 1)
            if overlap:
                for index in range(following, end):
                    if text[index - 1].isspace() and not text[index].isspace():
                        following = index
                        break
            start = following
    trimmed = []
    for start, end in spans:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start and (start, end) not in trimmed:
            trimmed.append((start, end))
    return trimmed


def digest(text):
    """Identity of the exact embedding input; equal digests reuse the stored vector."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def documents(items, config):
    """Turn texts or {id, text, metadata} objects into chunk records with deterministic ids."""
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_DOCUMENTS:
        fail("input", f"An upsert needs 1–{MAX_DOCUMENTS} documents.")
    chunks, sources, seen = [], [], set()
    for item in items:
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict) or set(item) - {"id", "text", "metadata"}:
            fail("input", "Each document is a string or an object with id, text and metadata only; put other fields in metadata.")
        text = text_value(item.get("text"), MAX_DOCUMENT_CHARS)
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            fail("input", "metadata must be an object.")
        checked(metadata, MAX_METADATA_BYTES, depth=8, label="metadata")
        if "id" in item:
            source = item["id"]
            if not isinstance(source, str) or not SOURCE_ID.fullmatch(source):
                fail("input", "Document ids need 1–200 visible characters without spaces or #.")
        else:
            source = "sha256:" + digest(text)[:32]
        if source in seen:
            fail("input", f"Duplicate document id {source!r} in one message (identical texts produce identical ids).")
        seen.add(source)
        sources.append(source)
        if not config["chunk_size"] and len(text) > config["max_text_chars"]:
            fail("input", f"A text exceeds max_text_chars ({config['max_text_chars']}); enable chunking with chunk_size.")
        spans = chunk_spans(text, config["chunk_size"], config["chunk_overlap"])
        for index, (start, end) in enumerate(spans):
            part = text[start:end]
            chunks.append({"id": source if len(spans) == 1 else f"{source}#{index}", "source_id": source,
                           "text": part, "metadata": metadata, "digest": digest(config["document_prefix"] + part),
                           "chunk": {"index": index, "count": len(spans), "start": start, "end": end}})
        if len(chunks) > MAX_CHUNKS:
            fail("input", f"One message may produce at most {MAX_CHUNKS} chunks.")
    return chunks, sources


def query_request(request, config):
    """Query text plus optional top_k, min_score and metadata equality filter."""
    if set(request) - {"operation", "text", "top_k", "min_score", "filter"}:
        fail("input", "A query accepts text, top_k, min_score and filter only.")
    text = text_value(request.get("text"), config["max_text_chars"], "Query text")
    top_k = request.get("top_k", config["top_k"])
    if type(top_k) is not int or not 1 <= top_k <= 100:
        fail("input", "top_k must be an integer between 1 and 100.")
    min_score = request.get("min_score", config["min_score"])
    if type(min_score) not in (int, float) or not math.isfinite(min_score) or not -1 <= min_score <= 1:
        fail("input", "min_score must be a number between -1 and 1.")
    return {"text": text, "top_k": top_k, "min_score": float(min_score), "filter": metadata_filter(request.get("filter"))}


def delete_request(request):
    """Exactly one selector: ids (documents or chunks), a metadata filter, or all."""
    selectors = [key for key in ("ids", "filter", "all") if key in request]
    if set(request) - {"operation", "ids", "filter", "all"} or len(selectors) != 1:
        fail("input", "delete needs exactly one of ids, filter or all.")
    if "ids" in request:
        ids = request["ids"]
        if not isinstance(ids, list) or not 1 <= len(ids) <= MAX_DELETE_IDS or any(
                not isinstance(item, str) or not ITEM_ID.fullmatch(item) for item in ids):
            fail("input", f"ids must list 1–{MAX_DELETE_IDS} document or chunk ids.")
        return {"ids": list(dict.fromkeys(ids))}
    if "filter" in request:
        return {"filter": metadata_filter(request["filter"], allow_empty=False)}
    if request["all"] is not True:
        fail("input", "all must be true.")
    return {"all": True}


def message(value, port, config):
    """Classify one input message as (operation, normalized request)."""
    parsed, structured = decode(value, config["max_input_bytes"])
    if port == "documents":
        if not structured or isinstance(parsed, str):
            return "upsert", documents([parsed], config)
        if isinstance(parsed, list):
            return "upsert", documents(parsed, config)
        if not isinstance(parsed, dict):
            fail("input", "documents accepts text, a document object, an array or an operation envelope.")
        operation = parsed.get("operation", "upsert" if "documents" in parsed else None)
        if operation is None:
            return "upsert", documents([parsed], config)
        if operation == "upsert":
            if set(parsed) - {"operation", "documents"} or "documents" not in parsed:
                fail("input", "An upsert envelope contains operation and documents only.")
            return "upsert", documents(parsed["documents"], config)
        if operation == "delete":
            return "delete", delete_request(parsed)
        if operation == "stats":
            if set(parsed) != {"operation"}:
                fail("input", "stats takes no other field.")
            return "stats", {}
        fail("input", "documents operations are upsert, delete and stats.")
    if port == "query":
        if not structured or isinstance(parsed, str):
            return "query", query_request({"text": parsed}, config)
        if not isinstance(parsed, dict):
            fail("input", "query accepts text or an object with text, top_k, min_score and filter.")
        operation = parsed.get("operation", "query")
        if operation == "stats":
            if set(parsed) != {"operation"}:
                fail("input", "stats takes no other field.")
            return "stats", {}
        if operation != "query":
            fail("input", "query operations are query and stats.")
        return "query", query_request(parsed, config)
    fail("input", "Unknown input port.")
