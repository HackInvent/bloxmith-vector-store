"""Node-private SQLite collections: float32 vectors, exact model binding, atomic writes.

numpy is imported lazily for the similarity scan; without it a pure-Python scan
serves small collections and larger ones fail with an explicit dependency error.
"""

from array import array
from contextlib import contextmanager
import fcntl
import hashlib
import heapq
import json
import math
from operator import mul
import os
from pathlib import Path
import sqlite3
import stat
import sys
import time

from .logic import VectorError, checked, fail, scalar_key

SCHEMA_VERSION = 1
MAX_COLLECTIONS = 16
PYTHON_SCAN_LIMIT = 8000000
SCAN_BATCH = 1024
LOCK_NAME = ".vector-store.lock"


def numpy_module():
    """Return numpy when importable; a missing or broken install selects the Python scan."""
    try:
        import numpy  # noqa: PLC0415 - optional dependency, imported only for a search.
    except Exception:  # noqa: BLE001 - any import failure means "not available".
        return None
    return numpy


def check(deadline, cancel):
    """One deadline and one Stop check cover locks, scans and writes."""
    if cancel():
        raise VectorError("cancelled", "Stopped before the collection operation finished; nothing was stored.")
    if time.monotonic() >= deadline:
        raise VectorError("timeout", "The collection operation exceeded timeout_sec; nothing was stored.", retryable=True)


def collection_path(directory, name):
    """The collection name never becomes a path component; its hash does."""
    return Path(directory) / ("vectors-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:32] + ".sqlite3")


def _storage(directory):
    directory = Path(directory)
    if not directory.is_absolute() or directory.resolve() != directory or not directory.is_dir():
        fail("storage", "Canonical node storage is required.")
    return directory


def _check_files(path, limit):
    for suffix in ("", "-journal", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.is_symlink():
            fail("storage", "Collection files must not be symbolic links.")
        if candidate.exists():
            info = candidate.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
                fail("storage", "Collection files must be private regular files within max_database_mb.")


def _create(directory, path, limit, deadline, cancel):
    """Serialize creation so concurrent workers cannot exceed the per-node collection cap."""
    lock = os.open(directory / LOCK_NAME, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        info = os.fstat(lock)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            fail("storage", "A private regular storage lock file is required.")
        while True:
            check(deadline, cancel)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.02)
        _check_files(path, limit)
        if not path.exists():
            if len(list(directory.glob("vectors-*.sqlite3"))) >= MAX_COLLECTIONS:
                fail("capacity", f"This node already holds {MAX_COLLECTIONS} collections; delete one before creating another.")
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
    finally:
        os.close(lock)


def _schema(db, name):
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        if db.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]:
            fail("storage", "Unknown database in the collection file; nothing was changed.")
        db.execute("CREATE TABLE meta (id INTEGER PRIMARY KEY CHECK(id=1), collection TEXT NOT NULL, model TEXT, "
                   "dimension INTEGER, endpoint TEXT, revision INTEGER NOT NULL)")
        db.execute("CREATE TABLE items (id TEXT PRIMARY KEY, source_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, "
                   "chunk_count INTEGER NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL, text TEXT NOT NULL, "
                   "digest TEXT NOT NULL, metadata TEXT NOT NULL, vector BLOB NOT NULL, revision INTEGER NOT NULL)")
        db.execute("CREATE INDEX items_source ON items(source_id)")
        db.execute("CREATE TABLE item_meta (item_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, "
                   "PRIMARY KEY(item_id, key))")
        db.execute("CREATE INDEX item_meta_value ON item_meta(key, value)")
        db.execute("INSERT INTO meta VALUES(1, ?, NULL, NULL, NULL, 0)", (name,))
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    elif version != SCHEMA_VERSION:
        fail("storage", "Unsupported collection schema; no migration or reset was attempted.")
    row = db.execute("SELECT collection FROM meta WHERE id=1").fetchone()
    if not row or row[0] != name:
        fail("storage", "The collection file does not match the configured collection.")


@contextmanager
def opened(directory, config, deadline, cancel, *, write):
    """Yield a connection inside one transaction, or None for a read of a missing collection."""
    directory = _storage(directory)
    path = collection_path(directory, config["collection"])
    limit = config["max_database_mb"] * 1048576
    if write:
        _create(directory, path, limit, deadline, cancel)
    else:
        _check_files(path, limit)
        if not path.exists() or path.stat().st_size == 0:
            yield None
            return
    check(deadline, cancel)
    if write:
        db = sqlite3.connect(path, timeout=0.1, isolation_level=None)
    else:
        # Not mode=ro: a read connection must still be able to roll back a hot journal left by a crash.
        db = sqlite3.connect(path, timeout=0.1, isolation_level=None)
        db.execute("PRAGMA query_only=ON")
    try:
        if hasattr(db, "enable_load_extension"):
            db.enable_load_extension(False)
        db.execute("PRAGMA busy_timeout=100")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA max_page_count=" + str(limit // db.execute("PRAGMA page_size").fetchone()[0]))
        db.set_progress_handler(lambda: int(bool(cancel()) or time.monotonic() >= deadline), 2000)
        while True:
            check(deadline, cancel)
            try:
                db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                    raise
                time.sleep(0.02)
        if write:
            _schema(db, config["collection"])
        elif db.execute("PRAGMA user_version").fetchone()[0] == 0:
            db.execute("ROLLBACK")
            yield None
            return
        else:
            _schema(db, config["collection"])
        yield db
        check(deadline, cancel)
        db.execute("COMMIT" if write else "ROLLBACK")
    except sqlite3.OperationalError as exc:
        _rollback(db)
        if "interrupted" in str(exc).lower():
            check(deadline, cancel)
        if "full" in str(exc).lower():
            fail("capacity", "The collection reached max_database_mb; the whole operation was rolled back.")
        raise
    except BaseException:
        _rollback(db)
        raise
    finally:
        db.close()


def _rollback(db):
    try:
        db.set_progress_handler(None, 0)
        if db.in_transaction:
            db.execute("ROLLBACK")
    except sqlite3.Error:
        pass


def binding(db):
    """The model and dimension the collection was built with (None before the first vectors)."""
    if db is None:
        return {"model": None, "dimension": None, "endpoint": None, "revision": 0}
    model, dimension, endpoint, revision = db.execute(
        "SELECT model, dimension, endpoint, revision FROM meta WHERE id=1").fetchone()
    return {"model": model, "dimension": dimension, "endpoint": endpoint, "revision": revision}


def require_model(bound, config):
    """A collection built with one model is never searched or extended with another."""
    if bound["model"] is not None and bound["model"] != config["model"]:
        fail("model_mismatch", f"Collection {config['collection']!r} was built with model {bound['model']!r}, "
             f"not {config['model']!r}; use the same model or delete all items first.",
             collection_model=bound["model"], configured_model=config["model"])


def require_dimension(bound, dimension):
    if bound["dimension"] is not None and bound["dimension"] != dimension:
        fail("dimension_mismatch", f"The endpoint returned {dimension}-dimensional vectors; "
             f"the collection stores {bound['dimension']} dimensions.",
             collection_dimension=bound["dimension"], received_dimension=dimension)


def existing(db, ids):
    """Stored digests for these item ids, to skip re-embedding unchanged chunks."""
    found = {}
    if db is None:
        return found
    ids = list(ids)
    for start in range(0, len(ids), 500):
        part = ids[start:start + 500]
        rows = db.execute("SELECT id, digest FROM items WHERE id IN (" + ",".join("?" for _ in part) + ")", part)
        found.update(rows.fetchall())
    return found


def pack(vector):
    """L2-normalized little-endian float32 blob; a zero vector cannot be compared."""
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if not norm or not math.isfinite(norm):
        fail("provider_response", "The endpoint returned a zero or non-finite vector.")
    packed = array("f", (value / norm for value in vector))
    if sys.byteorder == "big":
        packed.byteswap()
    return packed.tobytes()


def unpack(blob):
    values = array("f")
    values.frombytes(blob)
    if sys.byteorder == "big":
        values.byteswap()
    return values


def _filter_sql(selected):
    clauses, args = [], []
    for key in sorted(selected or {}):
        clauses.append("items.id IN (SELECT item_id FROM item_meta WHERE key=? AND value=?)")
        args.extend([key, scalar_key(selected[key])])
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), args


def count(db, selected=None):
    if db is None:
        return 0
    clause, args = _filter_sql(selected)
    return db.execute("SELECT count(*) FROM items" + clause, args).fetchone()[0]


def _index_metadata(db, item_id, metadata):
    db.execute("DELETE FROM item_meta WHERE item_id=?", (item_id,))
    rows = [(item_id, key, scalar_key(value)) for key, value in metadata.items()
            if value is None or type(value) in (str, int, float, bool)]
    db.executemany("INSERT INTO item_meta VALUES(?,?,?)", rows)


def _remove(db, ids):
    for start in range(0, len(ids), 500):
        part = ids[start:start + 500]
        marks = ",".join("?" for _ in part)
        db.execute(f"DELETE FROM item_meta WHERE item_id IN ({marks})", part)
        db.execute(f"DELETE FROM items WHERE id IN ({marks})", part)


def upsert(db, chunks, embedded, config, endpoint_display):
    """Write one message atomically: new vectors, reused vectors, stale chunks removed, capacity checked."""
    bound = binding(db)
    require_model(bound, config)
    if embedded:
        require_dimension(bound, len(next(iter(embedded.values()))))
    stored = existing(db, [chunk["id"] for chunk in chunks])
    for chunk in chunks:
        if chunk["id"] not in embedded and stored.get(chunk["id"]) != chunk["digest"]:
            fail("concurrent_change", "The collection changed while embedding; send the message again.", retryable=True)
    revision = bound["revision"] + 1
    keep = {chunk["id"] for chunk in chunks}
    stale = []
    for source in dict.fromkeys(chunk["source_id"] for chunk in chunks):
        stale.extend(row[0] for row in db.execute("SELECT id FROM items WHERE source_id=?", (source,))
                     if row[0] not in keep)
    _remove(db, stale)
    for chunk in chunks:
        info, metadata = chunk["chunk"], checked(chunk["metadata"], 1 << 20, label="metadata")
        if chunk["id"] in embedded:
            db.execute("INSERT INTO items VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                       "source_id=excluded.source_id, chunk_index=excluded.chunk_index, chunk_count=excluded.chunk_count, "
                       "start=excluded.start, end=excluded.end, text=excluded.text, digest=excluded.digest, "
                       "metadata=excluded.metadata, vector=excluded.vector, revision=excluded.revision",
                       (chunk["id"], chunk["source_id"], info["index"], info["count"], info["start"], info["end"],
                        chunk["text"], chunk["digest"], metadata, pack(embedded[chunk["id"]]), revision))
        else:
            db.execute("UPDATE items SET source_id=?, chunk_index=?, chunk_count=?, start=?, end=?, text=?, metadata=? "
                       "WHERE id=?", (chunk["source_id"], info["index"], info["count"], info["start"], info["end"],
                                      chunk["text"], metadata, chunk["id"]))
        _index_metadata(db, chunk["id"], chunk["metadata"])
    total = count(db)
    if total > config["max_items"]:
        fail("capacity", f"The collection would hold {total} items, above max_items ({config['max_items']}); "
             "the whole message was rolled back.")
    if embedded and bound["model"] is None:
        db.execute("UPDATE meta SET model=?, dimension=?, endpoint=? WHERE id=1",
                   (config["model"], len(next(iter(embedded.values()))), endpoint_display))
    db.execute("UPDATE meta SET revision=? WHERE id=1", (revision,))
    return {"embedded": len(embedded), "unchanged": len(chunks) - len(embedded), "removed_stale": len(stale),
            "items": total, "revision": revision}


def delete(db, request, config):
    """Remove ids (a document id removes all its chunks), a metadata selection, or everything."""
    if db is None:
        return {"deleted": 0, "items": 0, "revision": 0, "reset": bool(request.get("all"))}
    bound = binding(db)
    if request.get("all"):
        deleted = count(db)
        db.execute("DELETE FROM item_meta")
        db.execute("DELETE FROM items")
        db.execute("UPDATE meta SET model=NULL, dimension=NULL, endpoint=NULL, revision=? WHERE id=1", (bound["revision"] + 1,))
        return {"deleted": deleted, "items": 0, "revision": bound["revision"] + 1, "reset": True}
    if "ids" in request:
        ids, targets = request["ids"], []
        for start in range(0, len(ids), 400):
            part = ids[start:start + 400]
            marks = ",".join("?" for _ in part)
            targets.extend(row[0] for row in db.execute(
                f"SELECT id FROM items WHERE id IN ({marks}) OR source_id IN ({marks})", part + part))
    else:
        clause, args = _filter_sql(request["filter"])
        targets = [row[0] for row in db.execute("SELECT id FROM items" + clause, args)]
    targets = list(dict.fromkeys(targets))
    _remove(db, targets)
    revision = bound["revision"] + (1 if targets else 0)
    db.execute("UPDATE meta SET revision=? WHERE id=1", (revision,))
    return {"deleted": len(targets), "items": count(db), "revision": revision, "reset": False}


def statistics(db, config, path=None):
    bound = binding(db)
    items = count(db)
    sources = db.execute("SELECT count(DISTINCT source_id) FROM items").fetchone()[0] if db is not None else 0
    size = path.stat().st_size if path is not None and path.exists() else 0
    return {"model": bound["model"], "dimension": bound["dimension"], "configured_model": config["model"],
            "items": items, "sources": sources, "revision": bound["revision"], "database_bytes": size,
            "similarity": "numpy" if numpy_module() is not None else "python"}


def _top(candidates, top_k):
    """Highest scores first; equal scores ordered by id for deterministic results."""
    return heapq.nsmallest(top_k, candidates, key=lambda item: (-item[0], item[1]))


def search(db, query_vector, request, deadline, cancel, *, numpy="auto"):
    """Cosine similarity of the normalized query against every candidate; exact, bounded, cancellable."""
    selected = request["filter"]
    total = count(db, selected)
    if not total:
        return {"searched": 0, "items": [], "similarity": "none"}
    dimension = binding(db)["dimension"]
    if dimension != len(query_vector):
        require_dimension({"dimension": dimension}, len(query_vector))
    module = numpy_module() if numpy == "auto" else numpy
    if module is None and total * dimension > PYTHON_SCAN_LIMIT:
        fail("dependency", f"Searching {total} items of {dimension} dimensions needs numpy (pure-Python scans are "
             f"limited to {PYTHON_SCAN_LIMIT} values); install numpy in the Studio Python environment.")
    query = unpack(pack(query_vector))
    clause, args = _filter_sql(selected)
    cursor = db.execute("SELECT id, vector FROM items" + clause + " ORDER BY id", args)
    best, top_k, floor = [], request["top_k"], request["min_score"]
    if module is not None:
        target = module.frombuffer(query.tobytes(), dtype="<f4")
    while True:
        check(deadline, cancel)
        rows = cursor.fetchmany(SCAN_BATCH)
        if not rows:
            break
        if module is not None:
            matrix = module.frombuffer(b"".join(row[1] for row in rows), dtype="<f4").reshape(len(rows), dimension)
            scores = (matrix @ target).tolist()
        else:
            scores = [sum(map(mul, query, unpack(row[1]))) for row in rows]
        for row, score in zip(rows, scores):
            score = round(min(1.0, max(-1.0, float(score))), 6)
            if score >= floor:
                best.append((score, row[0]))
        best = _top(best, top_k)
    chosen = _top(best, top_k)
    items = []
    for score, item_id in chosen:
        row = db.execute("SELECT source_id, chunk_index, chunk_count, start, end, text, metadata FROM items WHERE id=?",
                         (item_id,)).fetchone()
        items.append({"id": item_id, "score": score, "text": row[5], "metadata": json.loads(row[6]),
                      "source_id": row[0], "chunk": {"index": row[1], "count": row[2], "start": row[3], "end": row[4]}})
    return {"searched": total, "items": items, "similarity": "numpy" if module is not None else "python"}
