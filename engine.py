"""One message, one operation: validate, embed outside any lock, then commit atomically."""

import time

from . import logic, provider, store
from .logic import VectorError

STORAGE_MESSAGE = "Persistent node storage is unavailable; run the graph from a saved project."


class Services:
    """Lazily resolved storage, wallet token and embedder shared by one activation."""

    def __init__(self, config, *, storage=None, resolve_secret=None, cancel=None, embedder=None):
        self.config = config
        self._storage, self._resolve, self._token = storage, resolve_secret, None
        self.cancel = cancel or (lambda: False)
        self._embedder = embedder

    def directory(self):
        if not callable(self._storage):
            raise VectorError("storage", STORAGE_MESSAGE)
        try:
            return self._storage()
        except Exception:  # noqa: BLE001 - scope and IO details stay out of public outputs.
            raise VectorError("storage", STORAGE_MESSAGE) from None

    def token(self):
        """Resolve the wallet reference only when a provider call is actually needed."""
        ref = self.config["api_key_ref"]
        if not ref:
            return ""
        if self._token is None:
            if not callable(self._resolve):
                raise VectorError("secret", "The wallet service is unavailable; api_key_ref cannot be resolved.")
            try:
                value = self._resolve(ref)
            except Exception:  # noqa: BLE001 - locked/missing wallet details are not exposed.
                raise VectorError("secret", "The wallet secret is unavailable: unlock the wallet and check api_key_ref.") from None
            if not isinstance(value, str) or not 1 <= len(value) <= 8192 or any(ord(c) < 33 or ord(c) > 126 for c in value):
                raise VectorError("secret", "The wallet secret must be a printable API key without spaces.")
            self._token = value
        return self._token

    def embed(self, texts, deadline):
        if self._embedder is not None:
            return self._embedder(texts, self.config, deadline=deadline, cancel=self.cancel)
        return provider.embed(texts, self.config, token=self.token(), cancel=self.cancel, deadline=deadline)


def _summary(config, bound):
    return {"collection": config["collection"], "model": bound.get("model") or config["model"],
            "dimension": bound.get("dimension")}


def upsert(chunks, sources, config, services, deadline):
    directory = services.directory()
    with store.opened(directory, config, deadline, services.cancel, write=False) as db:
        bound = store.binding(db)
        store.require_model(bound, config)
        known = store.existing(db, [chunk["id"] for chunk in chunks])
    pending = [chunk for chunk in chunks if known.get(chunk["id"]) != chunk["digest"]]
    embedded = {}
    if pending:
        vectors = services.embed([config["document_prefix"] + chunk["text"] for chunk in pending], deadline)
        if len(vectors) != len(pending):
            raise VectorError("provider_response", "The endpoint returned a different number of embeddings.")
        embedded = {chunk["id"]: vector for chunk, vector in zip(pending, vectors)}
    with store.opened(directory, config, deadline, services.cancel, write=True) as db:
        written = store.upsert(db, chunks, embedded, config, logic.endpoint(config)["display"])
        bound = store.binding(db)
    listed = sources[:1000]
    return {"result": {"operation": "upsert", **_summary(config, bound), "documents": len(sources),
                       "chunks": len(chunks), **written, "sources": listed, "sources_truncated": len(listed) < len(sources)}}


def query(request, config, services, deadline):
    directory = services.directory()
    with store.opened(directory, config, deadline, services.cancel, write=False) as db:
        bound = store.binding(db)
        store.require_model(bound, config)
        candidates = store.count(db, request["filter"])
    found = {"searched": 0, "items": [], "similarity": "none"}
    if candidates:
        vector = services.embed([config["query_prefix"] + request["text"]], deadline)[0]
        with store.opened(directory, config, deadline, services.cancel, write=False) as db:
            bound = store.binding(db)
            store.require_model(bound, config)
            found = store.search(db, vector, request, deadline, services.cancel)
    context = "\n\n---\n\n".join(item["text"] for item in found["items"])
    return {"matches": {**_summary(config, bound), "query": request["text"], "top_k": request["top_k"],
                        "min_score": request["min_score"], "filter": request["filter"], "searched": found["searched"],
                        "count": len(found["items"]), "items": found["items"], "context": context,
                        "similarity": found["similarity"]}}


def delete(request, config, services, deadline):
    if not config["allow_delete"]:
        raise VectorError("delete_disabled", "Deletion is disabled; check allow_delete in the settings to enable it.")
    directory = services.directory()
    path = store.collection_path(directory, config["collection"])
    if not path.exists():
        removed = {"deleted": 0, "items": 0, "revision": 0, "reset": bool(request.get("all"))}
        bound = {}
    else:
        with store.opened(directory, config, deadline, services.cancel, write=True) as db:
            removed = store.delete(db, request, config)
            bound = store.binding(db)
    selector = "all" if request.get("all") else ("ids" if "ids" in request else "filter")
    return {"result": {"operation": "delete", **_summary(config, bound), "selector": selector, **removed}}


def stats(config, services, deadline):
    directory = services.directory()
    path = store.collection_path(directory, config["collection"])
    with store.opened(directory, config, deadline, services.cancel, write=False) as db:
        figures = store.statistics(db, config, path)
    return {"result": {"operation": "stats", "collection": config["collection"], **figures}}


def process(value, port, config, services, deadline=None):
    """Return {port name: JSON value}; raises VectorError with the operation name attached."""
    deadline = min(deadline or float("inf"), time.monotonic() + config["timeout_sec"])
    operation = "query" if port == "query" else "upsert"
    try:
        operation, request = logic.message(value, port, config)
        store.check(deadline, services.cancel)
        if operation == "upsert":
            values = upsert(*request, config, services, deadline)
        elif operation == "query":
            values = query(request, config, services, deadline)
        elif operation == "delete":
            values = delete(request, config, services, deadline)
        else:
            values = stats(config, services, deadline)
        for name, item in values.items():
            logic.checked(item, logic.MAX_OUTPUT_BYTES, depth=40, label=name)
        return values
    except VectorError as exc:
        exc.operation = operation
        raise
