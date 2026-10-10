"""OpenAI-compatible /embeddings client: one configured endpoint, vetted addresses, bounded and cancellable.

Only the Python standard library is used. Each HTTP exchange runs in a short-lived
daemon thread so that Stop or the activation deadline can abort the socket at once.
"""

import http.client
import ipaddress
import json
import math
from pathlib import Path
import socket
import ssl
import threading
import time

from .logic import MAX_DIMENSION, VectorError, endpoint, fail

USER_AGENT = "BloxSmith-Vector-Store/0.1.0"
MAX_ERROR_BODY = 65536


def remaining(deadline):
    """Every socket operation shares the activation deadline."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise VectorError("timeout", "The embeddings request exceeded timeout_sec.", retryable=True)
    return max(0.001, left)


def _address(value):
    address = ipaddress.ip_address(str(value).split("%")[0])
    return getattr(address, "ipv4_mapped", None) or address


def address_allowed(value, config):
    """Loopback always; other addresses only with allow_remote, never link-local/metadata or multicast."""
    try:
        address = _address(value)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    if not config["allow_remote"]:
        return False
    return not (address.is_link_local or address.is_multicast or address.is_unspecified or address.is_reserved)


def provider_message(raw, token):
    """A short single-line provider explanation, with the credential removed if echoed."""
    text = raw.decode("utf-8", errors="replace")
    message = text
    try:
        document = json.loads(text)
        if isinstance(document, dict):
            error = document.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                message = error["message"]
            elif isinstance(error, str):
                message = error
            elif isinstance(document.get("message"), str):
                message = document["message"]
            elif isinstance(document.get("detail"), str):
                message = document["detail"]
    except ValueError:
        pass
    message = " ".join(str(message).split())
    if token:
        message = message.replace(token, "[redacted]")
    return message[:300]


def _status_error(status, raw, token):
    detail = {"http_status": status}
    explanation = provider_message(raw, token)
    if explanation:
        detail["provider_message"] = explanation
    if status in {401, 403}:
        return VectorError("auth", "The embeddings endpoint refused the credentials; check api_key_ref.", **detail)
    if status in {408, 429}:
        return VectorError("rate_limited", "The embeddings endpoint is rate limiting or timed out; retry later.",
                           retryable=True, **detail)
    if status >= 500:
        return VectorError("provider_unavailable", "The embeddings endpoint failed or is unavailable.",
                           retryable=True, **detail)
    if status in {400, 404, 413, 422}:
        return VectorError("provider_rejected", "The embeddings endpoint rejected the request; check base_url and model.",
                           **detail)
    return VectorError("http_status", "The embeddings endpoint returned an unexpected HTTP status.", **detail)


def _post(target, config, body, token, deadline, holder):
    """One exchange on a vetted numeric address; DNS is resolved once and never re-resolved."""
    host, port = target["host"], target["port"]
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
    except OSError:
        fail("network", "The embeddings host name could not be resolved.", retryable=True)
    if not infos or len(infos) > 32 or any(not address_allowed(item[4][0], config) for item in infos):
        fail("network", "The embeddings host resolves to an address this block may not contact; "
             "non-loopback servers need allow_remote.")
    local = all(_address(item[4][0]).is_loopback for item in infos)
    if target["scheme"] == "http" and not local:
        if not config["allow_insecure_http"]:
            fail("network", "Plain HTTP to a non-loopback address requires allow_insecure_http.")
        if token:
            fail("network", "The API key is never sent over plain HTTP to a non-loopback address.")
    context = None
    if target["scheme"] == "https":
        ca = config["ca_file"]
        if ca:
            path = Path(ca)
            if not path.is_file() or path.stat().st_size > 1048576:
                fail("network", "The configured ca_file is missing, not a file or too large.")
            context = ssl.create_default_context(cafile=str(path))
        else:
            context = ssl.create_default_context()
    sock = None
    for family, kind, proto, _, address in infos[:4]:
        candidate = socket.socket(family, kind, proto)
        holder["sock"] = candidate
        try:
            candidate.settimeout(remaining(deadline))
            candidate.connect(address)
            sock = candidate
            break
        except VectorError:
            candidate.close()
            raise
        except OSError:
            candidate.close()
    if sock is None:
        fail("network", "Unable to connect to the embeddings endpoint.", retryable=True)
    try:
        if context is not None:
            sock.settimeout(remaining(deadline))
            sock = context.wrap_socket(sock, server_hostname=host)
            holder["sock"] = sock
        connection = http.client.HTTPConnection(host, port, timeout=remaining(deadline))
        connection.sock = sock
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "Accept-Encoding": "identity", "User-Agent": USER_AGENT, "Connection": "close"}
        if token:
            headers["Authorization"] = "Bearer " + token
        sock.settimeout(remaining(deadline))
        connection.request("POST", target["path"], body=body, headers=headers)
        sock.settimeout(remaining(deadline))
        response = connection.getresponse()
        encoding = (response.getheader("Content-Encoding") or "identity").strip().lower()
        if encoding != "identity":
            fail("provider_response", "Compressed responses are not supported; identity encoding was requested.")
        if response.status != 200:
            raise _status_error(response.status, response.read(MAX_ERROR_BODY), token)
        limit = config["max_response_bytes"]
        length = response.getheader("Content-Length")
        if length is not None and (not length.strip().isdigit() or int(length) > limit):
            fail("provider_response", f"The embeddings response exceeds max_response_bytes ({limit}).")
        raw = response.read(limit + 1)
        if len(raw) > limit:
            fail("provider_response", f"The embeddings response exceeds max_response_bytes ({limit}).")
        try:
            return json.loads(raw.decode("utf-8"), parse_constant=lambda name: fail(
                "provider_response", "The embeddings response contains NaN or Infinity."))
        except VectorError:
            raise
        except (ValueError, UnicodeError, RecursionError):
            fail("provider_response", "The embeddings response is not valid JSON.")
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _abort(holder):
    """Unblock the worker thread; the half-finished exchange is discarded."""
    sock = holder.get("sock")
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass


def _call(target, config, body, token, cancel, deadline):
    """Run one exchange in a daemon thread and stop waiting on Stop or the deadline."""
    box, holder = {}, {}

    def work():
        try:
            box["value"] = _post(target, config, body, token, deadline, holder)
        except BaseException as exc:  # noqa: BLE001 - transported to the caller below.
            box["error"] = exc

    thread = threading.Thread(target=work, name="vector-store-embeddings", daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(0.05)
        if not thread.is_alive():
            break
        if cancel():
            _abort(holder)
            raise VectorError("cancelled", "Stopped while waiting for the embeddings endpoint; nothing was stored.")
        if time.monotonic() >= deadline:
            _abort(holder)
            raise VectorError("timeout", "The embeddings request exceeded timeout_sec; nothing was stored.", retryable=True)
    error = box.get("error")
    if isinstance(error, VectorError):
        raise error
    if isinstance(error, socket.timeout):
        raise VectorError("timeout", "The embeddings request exceeded timeout_sec; nothing was stored.", retryable=True)
    if isinstance(error, ssl.SSLError):
        raise VectorError("network", "TLS negotiation or certificate verification failed.")
    if isinstance(error, http.client.HTTPException):
        raise VectorError("provider_response", "The embeddings endpoint returned a malformed HTTP response.", retryable=True)
    if isinstance(error, OSError):
        raise VectorError("network", "The connection to the embeddings endpoint failed.", retryable=True)
    if error is not None:
        raise VectorError("internal", "The embeddings request failed unexpectedly.")
    return box["value"]


def vectors(payload, count):
    """Validate the OpenAI embeddings shape: one finite float array per input, in index order."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or len(data) != count:
        fail("provider_response", "The embeddings response does not contain one embedding per input text.")
    slots = [None] * count
    for position, item in enumerate(data):
        if not isinstance(item, dict) or "embedding" not in item:
            fail("provider_response", "An embeddings item has no embedding array.")
        index = item.get("index", position)
        if type(index) is not int or not 0 <= index < count or slots[index] is not None:
            fail("provider_response", "The embeddings response has missing or duplicate indexes.")
        values = item["embedding"]
        if isinstance(values, str):
            fail("provider_response", "Base64 embeddings are not supported; the endpoint must return float arrays.")
        if not isinstance(values, list) or not 1 <= len(values) <= MAX_DIMENSION:
            fail("provider_response", f"Each embedding must be an array of 1–{MAX_DIMENSION} numbers.")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            fail("provider_response", "Embeddings must contain finite numbers only.")
        slots[index] = [float(value) for value in values]
    if len({len(item) for item in slots}) != 1:
        fail("provider_response", "The embeddings in one response have different dimensions.")
    return slots


def embed(texts, config, *, token="", cancel=lambda: False, deadline=None):
    """Embed texts in batches of batch_size; any failure stores nothing and is never retried here."""
    target = endpoint(config)
    deadline = deadline if deadline is not None else time.monotonic() + config["timeout_sec"]
    result = []
    for start in range(0, len(texts), config["batch_size"]):
        if cancel():
            raise VectorError("cancelled", "Stopped before the embeddings request; nothing was stored.")
        batch = texts[start:start + config["batch_size"]]
        body = json.dumps({"model": config["model"], "input": batch}, ensure_ascii=False).encode("utf-8")
        result.extend(vectors(_call(target, config, body, token, cancel, deadline), len(batch)))
    if len({len(item) for item in result}) > 1:
        fail("dimension_mismatch", "The endpoint returned embeddings of different dimensions across batches.")
    return result
