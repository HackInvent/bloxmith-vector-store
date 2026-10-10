"""FB1/FB2/FB3/FB4/FB5/FB6/FB8: settings, parsing, chunking, storage, ranking and deletion without a server.

The provider is replaced by an explicit in-process embedder with the same signature;
the HTTP client itself is exercised against a loopback fixture in F5.02.
"""

from contextlib import contextmanager
import ipaddress
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(Path(__file__).resolve().parent)]
from blocs.vector_store import engine, logic, provider, store
from blocs.vector_store.block import VectorStoreBlock
from blocs.vector_store.logic import VectorError
from bloxsmith_app.block_api import BlockRuntimeContext
from bloxsmith_app.block_runtime import BlockInputEvent
from embeddings_fixture import DIMENSION, fake_embedder, vector

SECRET = "fixture-vector-key-7Qm"
DOCS = [{"id": "cats", "text": "Cats purr and sleep on the warm sofa", "metadata": {"lang": "en", "topic": "pets", "rank": 1}},
        {"id": "dogs", "text": "Dogs bark at the mail carrier every morning", "metadata": {"lang": "en", "topic": "pets", "rank": 2}},
        {"id": "tea", "text": "Green tea is brewed with water below boiling", "metadata": {"lang": "en", "topic": "food", "flag": True}},
        {"id": "chats", "text": "Les chats dorment sur le canapé chaud", "metadata": {"lang": "fr", "topic": "pets", "rank": 1.0}}]


def refused(call, code=None):
    """A refusal must be an explicit VectorError, never an empty success."""
    try:
        call()
    except VectorError as exc:
        if code:
            assert exc.code == code, (code, exc.code, str(exc))
        return exc
    raise AssertionError("Expected a VectorError refusal")


def cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


@contextmanager
def collection(**overrides):
    """A disposable node directory and an engine service bound to a counting fake embedder."""
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder).resolve()
        config = logic.configuration({**VectorStoreBlock().default_config(), **overrides})
        calls = []
        services = engine.Services(config, storage=lambda: directory, embedder=fake_embedder(calls=calls))
        yield SimpleNamespace(directory=directory, config=config, calls=calls, services=services,
                              run=lambda value, port="documents", cfg=None, svc=None: engine.process(
                                  value if isinstance(value, str) else json.dumps(value), port, cfg or config, svc or services))


def settings():
    # - FB1 - Settings are validated without network, wallet or storage access.
    defaults = VectorStoreBlock().default_config()
    config = logic.configuration(defaults)
    assert config["base_url"] == "http://127.0.0.1:11434/v1" and config["model"] == "nomic-embed-text"
    assert not config["allow_remote"] and not config["allow_delete"] and config["api_key_ref"] == ""
    assert logic.endpoint(config)["path"] == "/v1/embeddings"
    assert logic.endpoint({**config, "base_url": "http://localhost:1234/v1/"})["loopback"]
    assert logic.endpoint({**config, "base_url": "http://[::1]:8000"})["path"] == "/embeddings"
    good = [{"base_url": "https://api.example.test/v1", "allow_remote": True, "api_key_ref": "secret://workspace/emb_key"},
            {"base_url": "http://gpu-box.example.test:8000/v1", "allow_remote": True, "allow_insecure_http": True},
            {"collection": "docs.v2_en-1", "min_score": -1, "top_k": 100, "chunk_size": 0, "chunk_overlap": 50000},
            {"chunk_size": 100, "chunk_overlap": 50, "query_prefix": "search_query: ", "ca_file": "/etc/ssl/custom.pem"}]
    for raw in good:
        logic.configuration({**defaults, **raw})
    bad = [{"collection": ""}, {"collection": "../escape"}, {"collection": "a" * 65}, {"collection": 3},
           {"model": ""}, {"model": "two words"}, {"api_key_ref": "sk-live-raw-key-pasted"},
           {"api_key_ref": "secret://other/name"}, {"base_url": "ftp://127.0.0.1/v1"},
           {"base_url": "http://user:pass@127.0.0.1/v1"}, {"base_url": "http://127.0.0.1/v1?x=1"},
           {"base_url": "http://127.0.0.1/v1/../admin"}, {"base_url": "http://127.0.0.1:99999/v1"},
           {"base_url": "https://api.example.test/v1"}, {"base_url": "http://gpu-box.example.test/v1", "allow_remote": True},
           {"base_url": "http://gpu-box.example.test/v1", "allow_remote": True, "allow_insecure_http": True,
            "api_key_ref": "secret://workspace/emb_key"},
           {"base_url": "http://169.254.169.254/v1", "allow_remote": True}, {"base_url": "http://0.0.0.0/v1"},
           {"top_k": 0}, {"top_k": 101}, {"top_k": True}, {"top_k": 5.0}, {"min_score": 1.5}, {"min_score": "0.2"},
           {"min_score": float("nan")}, {"chunk_size": 50}, {"chunk_size": 1000, "chunk_overlap": 501},
           {"chunk_size": 9000, "max_text_chars": 8000}, {"allow_delete": "yes"}, {"allow_remote": 1},
           {"ca_file": "relative.pem"}, {"query_prefix": "line\nbreak"}, {"timeout_sec": 0}, {"max_items": 0},
           {"unknown_setting": 1}]
    for raw in bad:
        refused(lambda raw=raw: logic.configuration({**defaults, **raw}), "config")
    assert "never paste" in str(refused(lambda: logic.configuration({**defaults, "api_key_ref": "sk-raw"})))


def chunking():
    # - FB2 - Character chunks with overlap are deterministic, cut at whitespace and cover the whole text.
    text = " ".join(f"word{index:04d}" for index in range(900))
    spans = logic.chunk_spans(text, 1000, 100)
    assert spans == logic.chunk_spans(text, 1000, 100) and len(spans) > 4
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (start, end), (following, _) in zip(spans, spans[1:]):
        assert end - start <= 1000 and following < end and end - following <= 100
        assert text[following - 1] == " " and text[start] != " " and text[end - 1] != " "
    covered = set()
    for start, end in spans:
        covered.update(range(start, end))
    assert all(index in covered for index, char in enumerate(text) if char != " ")
    assert logic.chunk_spans("short text", 1000, 100) == [(0, 10)]
    assert logic.chunk_spans("  padded  ", 0, 0) == [(2, 8)]
    unbroken = "x" * 2500
    assert logic.chunk_spans(unbroken, 1000, 0) == [(0, 1000), (1000, 2000), (2000, 2500)]
    config = logic.configuration({**VectorStoreBlock().default_config(), "chunk_size": 1000, "chunk_overlap": 100})
    chunks, sources = logic.documents([{"id": "guide", "text": text, "metadata": {"page": 3}}], config)
    assert sources == ["guide"] and [chunk["id"] for chunk in chunks] == [f"guide#{i}" for i in range(len(spans))]
    assert all(chunk["source_id"] == "guide" and chunk["metadata"] == {"page": 3} for chunk in chunks)
    assert [chunk["chunk"]["start"] for chunk in chunks] == [start for start, _ in spans]
    assert chunks[0]["chunk"]["count"] == len(spans) and text[chunks[1]["chunk"]["start"]:chunks[1]["chunk"]["end"]] == chunks[1]["text"]
    single, _ = logic.documents(["One short text."], config)
    assert single[0]["id"] == single[0]["source_id"] == "sha256:" + logic.digest("One short text.")[:32]
    assert logic.documents(["One short text."], config)[0][0]["id"] == single[0]["id"]
    prefixed = logic.configuration({**config, "document_prefix": "search_document: "})
    assert logic.documents(["One short text."], prefixed)[0][0]["digest"] != single[0]["digest"]
    whole = logic.configuration({**config, "chunk_size": 0})
    refused(lambda: logic.documents(["x " * 5000], whole), "input")


def parsing():
    # - FB1 - Messages are classified strictly; malformed input is refused before any side effect.
    config = logic.configuration(VectorStoreBlock().default_config())
    assert logic.message("Plain text to index", "documents", config)[0] == "upsert"
    refused(lambda: logic.message("[draft] notes", "documents", config), "input")
    refused(lambda: logic.message("[draft] notes", "query", config), "input")
    assert logic.message('["a", {"text": "b", "metadata": {"k": 1}}]', "documents", config)[1][1][0].startswith("sha256:")
    assert logic.message({"documents": [{"id": "x", "text": "y"}]}, "documents", config)[1][1] == ["x"]
    assert logic.message({"operation": "upsert", "documents": ["y"]}, "documents", config)[0] == "upsert"
    assert logic.message({"operation": "stats"}, "documents", config) == ("stats", {})
    assert logic.message('{"operation":"stats"}', "query", config) == ("stats", {})
    assert logic.message({"operation": "delete", "ids": ["a", "a", "b#1"]}, "documents", config) == ("delete", {"ids": ["a", "b#1"]})
    assert logic.message({"operation": "delete", "filter": {"topic": "pets"}}, "documents", config)[1] == {"filter": {"topic": "pets"}}
    assert logic.message({"operation": "delete", "all": True}, "documents", config)[1] == {"all": True}
    query = logic.message({"text": "cats", "top_k": 2, "min_score": 0.1, "filter": {"lang": "en", "rank": 1}}, "query", config)[1]
    assert query == {"text": "cats", "top_k": 2, "min_score": 0.1, "filter": {"lang": "en", "rank": 1}}
    assert logic.message("Why do cats purr?", "query", config)[1]["top_k"] == 5
    for value, port in (('{"text": "a", "text": "b"}', "documents"), ('{"text": NaN}', "documents"), ("{broken", "query"),
                        ({"id": "x", "text": "y", "title": "extra"}, "documents"), ({"id": "has space", "text": "y"}, "documents"),
                        ({"id": "has#hash", "text": "y"}, "documents"), ({"text": "   "}, "documents"), (["same", "same"], "documents"),
                        ([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}], "documents"), ({"text": "y", "metadata": []}, "documents"),
                        ({"text": "y", "metadata": {"blob": "x" * 9000}}, "documents"), ([], "documents"), (42, "documents"),
                        ({"operation": "delete"}, "documents"), ({"operation": "delete", "ids": [], "all": True}, "documents"),
                        ({"operation": "delete", "filter": {}}, "documents"), ({"operation": "delete", "all": "yes"}, "documents"),
                        ({"operation": "drop"}, "documents"), ({"operation": "stats", "x": 1}, "query"),
                        ({"operation": "delete", "ids": ["a"]}, "query"), ({"text": "a", "top_k": 0}, "query"),
                        ({"text": "a", "min_score": 2}, "query"), ({"text": "a", "filter": {"k": [1]}}, "query"),
                        ({"text": "a", "filter": {"k": {"nested": 1}}}, "query"), ({"text": "a", "extra": 1}, "query"),
                        ("x" * 9000, "query"), ({"text": "a"}, "other")):
        refused(lambda value=value, port=port: logic.message(value, port, config), "input")
    small = logic.configuration({**config, "max_input_bytes": 1024})
    refused(lambda: logic.message("x" * 2000, "documents", small), "input")
    assert logic.scalar_key(1) == logic.scalar_key(1.0) != logic.scalar_key(True) != logic.scalar_key("1")


def storage_and_ranking():
    # - FB3 - Unchanged chunks reuse stored vectors; changed and stale chunks are handled atomically.
    with collection() as case:
        first = case.run(DOCS)["result"]
        assert first["embedded"] == 4 and first["items"] == 4 and first["dimension"] == DIMENSION
        assert len(case.calls) == 1 and len(case.calls[0]["inputs"]) == 4
        again = case.run(DOCS)["result"]
        assert again["embedded"] == 0 and again["unchanged"] == 4 and len(case.calls) == 1
        relabeled = case.run([{**DOCS[0], "metadata": {"lang": "en", "topic": "cats"}}])["result"]
        assert relabeled["embedded"] == 0 and len(case.calls) == 1
        changed = case.run([{**DOCS[0], "text": "Cats purr loudly on the warm sofa"}, DOCS[1]])["result"]
        assert changed["embedded"] == 1 and changed["unchanged"] == 1 and case.calls[-1]["inputs"] == ["Cats purr loudly on the warm sofa"]
        long_text = " ".join(f"chapter{index} cats and dogs" for index in range(400))
        chunked = case.run({"id": "book", "text": long_text})["result"]
        assert chunked["chunks"] > 3 and chunked["items"] == 4 + chunked["chunks"]
        shorter = case.run({"id": "book", "text": "A short replacement about tea"})["result"]
        assert shorter["removed_stale"] == chunked["chunks"] and shorter["items"] == 5
        # - FB4 - Exact cosine ranking, deterministic ties, min_score, top_k and metadata equality filters.
        case.run({"id": "book", "text": long_text})
        question = "Do cats sleep on the sofa?"
        found = case.run(question, "query")["matches"]
        with store.opened(case.directory, case.config, time.monotonic() + 30, lambda: False, write=False) as db:
            rows = db.execute("SELECT id, text FROM items").fetchall()
        exact = {ident: cosine(vector(question), vector(text)) for ident, text in rows}
        everything = case.run({"text": question, "top_k": 100, "min_score": -1}, "query")["matches"]["items"]
        assert {item["id"] for item in everything} == set(exact) and len(everything) == len(rows)
        assert all(abs(item["score"] - exact[item["id"]]) <= 2e-6 for item in everything), "float32 cosine must match"
        assert [(item["score"], item["id"]) for item in everything] == sorted(
            ((item["score"], item["id"]) for item in everything), key=lambda pair: (-pair[0], pair[1])), "score then id order"
        expected = [item["id"] for item in everything if item["score"] >= 0][:5]
        assert [item["id"] for item in found["items"]] == expected
        assert found["items"][0]["id"] == "cats" and found["count"] == 5 and found["searched"] == len(rows)
        assert found["context"].startswith("Cats purr loudly") and "\n\n---\n\n" in found["context"]
        top = found["items"][0]
        assert top["metadata"] == {"lang": "en", "topic": "pets", "rank": 1} and top["chunk"] == {"index": 0, "count": 1, "start": 0, "end": 33}
        assert [item["id"] for item in case.run({"text": question, "top_k": 2}, "query")["matches"]["items"]] == expected[:2]
        assert case.run({"text": question, "min_score": 0.99}, "query")["matches"]["items"] == []
        filtered = case.run({"text": question, "filter": {"lang": "fr"}}, "query")["matches"]
        assert [item["id"] for item in filtered["items"]] == ["chats"] and filtered["searched"] == 1
        assert {item["id"] for item in case.run({"text": "cats", "filter": {"rank": 1}}, "query")["matches"]["items"]} == {"cats", "chats"}
        assert [item["id"] for item in case.run({"text": "tea", "filter": {"flag": True}}, "query")["matches"]["items"]] == ["tea"]
        assert case.run({"text": "tea", "filter": {"flag": 1}}, "query")["matches"]["items"] == []
        calls = len(case.calls)
        nothing = case.run({"text": "cats", "filter": {"topic": "absent"}}, "query")["matches"]
        assert nothing["items"] == [] and nothing["searched"] == 0 and len(case.calls) == calls, "Empty selections skip the provider"
        # Python fallback and numpy agree; a too-large scan without numpy fails explicitly.
        with store.opened(case.directory, case.config, time.monotonic() + 30, lambda: False, write=False) as db:
            request = {"top_k": 5, "min_score": 0.0, "filter": {}}
            fast = store.search(db, vector(question), request, time.monotonic() + 30, lambda: False)
            slow = store.search(db, vector(question), request, time.monotonic() + 30, lambda: False, numpy=None)
            assert slow["similarity"] == "python" and [i["id"] for i in fast["items"]] == [i["id"] for i in slow["items"]]
            assert all(abs(a["score"] - b["score"]) <= 2e-6 for a, b in zip(fast["items"], slow["items"]))
            with patch.object(store, "PYTHON_SCAN_LIMIT", 10):
                refused(lambda: store.search(db, vector(question), request, time.monotonic() + 30, lambda: False, numpy=None),
                        "dependency")
            refused(lambda: store.search(db, vector(question), request, time.monotonic() + 30, lambda: True), "cancelled")
        with patch.dict(sys.modules, {"numpy": None}):
            assert store.numpy_module() is None
            assert case.run(question, "query")["matches"]["similarity"] == "python"
        stats = case.run({"operation": "stats"})["result"]
        assert stats["items"] == stats["sources"] - 1 + chunked["chunks"] and stats["model"] == "nomic-embed-text"
        assert len(case.calls) == calls + 1, "stats never calls the provider"
        # - FB3 - One model and one dimension per collection; capacity and mismatches change nothing.
        other = logic.configuration({**case.config, "model": "other-model"})
        mismatch = refused(lambda: case.run("cats", "query", other), "model_mismatch")
        assert mismatch.details["collection_model"] == "nomic-embed-text"
        refused(lambda: case.run(["new text"], "documents", other), "model_mismatch")
        small = engine.Services(case.config, storage=lambda: case.directory, embedder=fake_embedder(dimension=8))
        refused(lambda: case.run(["a brand new text"], svc=small), "dimension_mismatch")
        refused(lambda: case.run("cats", "query", None, small), "dimension_mismatch")
        before = case.run({"operation": "stats"})["result"]["items"]
        tight = logic.configuration({**case.config, "max_items": before + 1})
        refused(lambda: case.run(["fresh one", "fresh two"], "documents", tight), "capacity")
        assert case.run({"operation": "stats"})["result"]["items"] == before
        # - FB5 - Deletion is opt-in and selects ids (whole documents), filters or everything.
        refused(lambda: case.run({"operation": "delete", "ids": ["cats"]}), "delete_disabled")
        allowed = logic.configuration({**case.config, "allow_delete": True})
        removed = case.run({"operation": "delete", "ids": ["book", "missing"]}, "documents", allowed)["result"]
        assert removed["deleted"] == chunked["chunks"] and removed["items"] == 4 and not removed["reset"]
        assert case.run({"operation": "delete", "ids": ["dogs"]}, "documents", allowed)["result"]["deleted"] == 1
        assert case.run({"operation": "delete", "filter": {"lang": "fr"}}, "documents", allowed)["result"]["deleted"] == 1
        everything = case.run({"operation": "delete", "all": True}, "documents", allowed)["result"]
        assert everything["deleted"] == 2 and everything["items"] == 0 and everything["reset"]
        rebound = case.run(["now another model"], "documents", other)["result"]
        assert rebound["model"] == "other-model" and rebound["embedded"] == 1


def limits_and_storage():
    # - FB3 - Deadlines, Stop, unsafe files and collection caps are explicit failures.
    with collection() as case:
        cancelled = engine.Services(case.config, storage=lambda: case.directory, embedder=fake_embedder(), cancel=lambda: True)
        refused(lambda: case.run(["a"], svc=cancelled), "cancelled")
        assert not any(case.directory.iterdir()), "A cancelled message must not create files"
        refused(lambda: engine.process("a", "documents", case.config, case.services, deadline=time.monotonic() - 1), "timeout")
        case.run(["first"])
        path = store.collection_path(case.directory, "default")
        assert path.is_file() and oct(path.stat().st_mode & 0o777) == "0o600"
        os.replace(path, case.directory / "elsewhere.sqlite3")
        os.symlink(case.directory / "elsewhere.sqlite3", path)
        refused(lambda: case.run(["second"]), "storage")
        refused(lambda: case.run("first", "query"), "storage")
        path.unlink()
        foreign = sqlite3.connect(path)
        foreign.execute("CREATE TABLE other(x)")
        foreign.commit()
        foreign.close()
        refused(lambda: case.run(["second"]), "storage")
        path.unlink()
        for index in range(store.MAX_COLLECTIONS):
            store.collection_path(case.directory, f"c{index}").write_bytes(b"")
        refused(lambda: case.run(["third"]), "capacity")
        missing = engine.Services(case.config, storage=None, embedder=fake_embedder())
        refused(lambda: case.run(["x"], svc=missing), "storage")
        broken = engine.Services(case.config, storage=lambda: (_ for _ in ()).throw(RuntimeError("/private/path")))
        assert "/private/path" not in str(refused(lambda: case.run(["x"], svc=broken), "storage"))


def secrets_and_policy():
    # - FB6 - The key comes only from the wallet, lazily, and never appears in errors; addresses are vetted.
    config = logic.configuration({**VectorStoreBlock().default_config(), "api_key_ref": "secret://workspace/emb"})
    asked = []
    services = engine.Services(config, resolve_secret=lambda ref: asked.append(ref) or SECRET)
    assert not asked and services.token() == SECRET and services.token() == SECRET and asked == ["secret://workspace/emb"]
    locked = engine.Services(config, resolve_secret=lambda ref: (_ for _ in ()).throw(RuntimeError("wallet locked " + SECRET)))
    error = refused(locked.token, "secret")
    assert SECRET not in json.dumps(error.envelope())
    refused(engine.Services(config, resolve_secret=lambda ref: "bad key\nwith newline").token, "secret")
    assert engine.Services(logic.configuration(VectorStoreBlock().default_config())).token() == ""
    assert "[redacted]" in provider.provider_message(json.dumps({"error": {"message": "bad " + SECRET}}).encode(), SECRET)
    assert SECRET not in provider.provider_message(json.dumps({"error": {"message": "bad " + SECRET}}).encode(), SECRET)
    assert provider.provider_message(b"x" * 1000, "")[:5] == "xxxxx" and len(provider.provider_message(b"x" * 1000, "")) == 300
    local = logic.configuration(VectorStoreBlock().default_config())
    remote = logic.configuration({**local, "allow_remote": True, "base_url": "https://api.example.test/v1"})
    for address in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
        assert provider.address_allowed(address, local) and provider.address_allowed(address, remote)
    # Private ranges are built numerically: tests never name a concrete LAN coordinate.
    for network in (ipaddress.ip_network((10 << 24, 8)), ipaddress.ip_network(((192 << 24) | (168 << 16), 16)),
                    ipaddress.ip_network("fc00::/7")):
        address = str(network[1])
        assert not provider.address_allowed(address, local) and provider.address_allowed(address, remote)
    for address in ("169.254.169.254", "fe80::1", "224.0.0.1", "0.0.0.0", "not-an-address"):
        assert not provider.address_allowed(address, remote)
    assert provider.vectors({"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]}, 2) == [[1.0, 0.0], [0.0, 1.0]]
    for payload in ({"data": []}, {"data": [{"index": 0, "embedding": "AAAA"}]}, {"data": [{"index": 0, "embedding": [1, True]}]},
                    {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [1.0]}]},
                    {"data": [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [1.0, 2.0]}]}, []):
        refused(lambda payload=payload: provider.vectors(payload, 2 if isinstance(payload, dict) and len(payload.get("data", [])) == 2 else 1),
                "provider_response")
    refused(lambda: store.pack([0.0, 0.0]), "provider_response")


def block_contract():
    # - FB7 - Direct execute_runtime: skipped without input, mutations before queries, errors routed, no leakage.
    block = VectorStoreBlock()
    ports_in = (SimpleNamespace(id=1, name="documents"), SimpleNamespace(id=2, name="query"))
    ports_out = (SimpleNamespace(id=1, name="matches"), SimpleNamespace(id=2, name="result"), SimpleNamespace(id=3, name="error"))

    def event(port, value):
        return BlockInputEvent(edge_id=f"e{port}", input_port_id=port, input_port_name="", source_node_id="s",
                               source_port_id=1, value=value)

    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder).resolve()
        services = {"get_block_storage_dir": lambda: directory, "cancel_requested": lambda: False}
        assert block.execute_runtime(BlockRuntimeContext(config={}, input_ports=ports_in, output_ports=ports_out,
                                                         services=services)).status == "skipped"
        assert block.initialize_runtime(BlockRuntimeContext(config={}, services={})).status == "failed"
        assert block.initialize_runtime(BlockRuntimeContext(config={}, services=services)).status == "success"
        bad = block.execute_runtime(BlockRuntimeContext(config={"top_k": 0}, input_ports=ports_in, output_ports=ports_out,
                                                        input_events=(event(1, "x"),), services=services))
        assert bad.status == "failed" and "top_k" in bad.error and not bad.outputs
        refused(lambda: block.prepare_runtime(BlockRuntimeContext(config={"base_url": "http://user:pw@127.0.0.1/v1"})), "config")
        with patch.object(engine.Services, "embed", lambda self, texts, deadline: [vector(text) for text in texts]):
            both = block.execute_runtime(BlockRuntimeContext(
                config={}, input_ports=ports_in, output_ports=ports_out, services=services,
                input_events=(event(2, "Which pets purr?"), event(1, json.dumps(DOCS)))))
            values = {output.port_name: json.loads(output.value) for output in both.outputs}
            assert both.status == "success" and set(values) == {"matches", "result"}, values
            assert values["result"]["embedded"] == 4 and values["matches"]["items"][0]["id"] == "cats"
            assert both.metadata["vector_store"]["operations"] == ["upsert", "query"]
            doubled = block.execute_runtime(BlockRuntimeContext(
                config={}, input_ports=ports_in, output_ports=ports_out, services=services,
                input_events=(event(2, "a"), event(2, "b"), event(1, "[broken"))))
            error = json.loads(next(o.value for o in doubled.outputs if o.port_name == "error"))
            assert doubled.status == "success" and [o.port_name for o in doubled.outputs] == ["error"]
            assert error["code"] == "batch" and error["input"] == "query"
            follow = error["additional_errors"][0]
            assert follow["code"] == "input" and follow["input"] == "documents" and follow["operation"] == "upsert"
            manual = block.execute_runtime(BlockRuntimeContext(config={}, inputs={"query": "cats", "2": "cats"},
                                                               input_ports=ports_in, output_ports=ports_out, services=services))
            assert json.loads(manual.outputs[0].value)["count"] >= 1
        keyed = {"api_key_ref": "secret://workspace/emb"}
        leaked = block.execute_runtime(BlockRuntimeContext(
            config=keyed, input_ports=ports_in, output_ports=ports_out, input_events=(event(1, "new text"),),
            services={**services, "resolve_secret": lambda ref: (_ for _ in ()).throw(RuntimeError(SECRET)),
                      "redact_secrets": lambda text: str(text).replace(SECRET, "[redacted]")}))
        assert json.loads(leaked.outputs[0].value)["code"] == "secret" and SECRET not in json.dumps([o.value for o in leaked.outputs] + leaked.logs)
    # - FB8 - Apply validates the draft and replaces the settings exactly; a pasted key is refused.
    node = block.build_node_payload(node_id="vs", position={"x": 0, "y": 0})
    refusal = block.handle_ui_action(node=node, action="save_settings", values={"node_patch": {"config": {"api_key_ref": "sk-raw"}}})
    assert "secret://workspace/" in refusal["error"]
    accepted = block.handle_ui_action(node=node, action="save_settings", values={"node_patch": {"config": {"top_k": 7}}})
    operation = accepted["graph_operations"][-1]
    assert operation["op"] == "update_node_config" and operation["replace"] and operation["config"]["top_k"] == 7
    for render in (block.render_modal, block.render_inspector_panel, block.render_node_card):
        html = render(node={**node, "config": {**node["config"], "api_key_ref": "secret://workspace/emb"}})["html"]
        assert "{{" not in html and ("secret://workspace/emb" not in html or render is not block.render_node_card)
    card = block.render_node_card(node={**node, "config": {**node["config"], "base_url": "https://api.example.test/v1"}})["html"]
    assert "default · nomic-embed-text" in card and "api.example.test" not in card
    mini = block.render_mini_node_card(node={**node, "config": {**node["config"], "api_key_ref": "secret://workspace/emb"}})["html"]
    assert "data-vector-store-mini-card" in mini and "default · nomic-embed-text" in mini and "secret://" not in mini
    english = json.loads((Path(block.directory) / "locales/en.json").read_text(encoding="utf-8"))
    french = json.loads((Path(block.directory) / "locales/fr.json").read_text(encoding="utf-8"))
    assert set(english) == set(french) and english["block.vector_store.model.description"] == block.model["description"]
    fields = json.loads((Path(block.directory) / "fields.json").read_text(encoding="utf-8"))
    assert [field["key"] for field in fields] == list(block.default_config())
    assert all(english[f"block.vector_store.help_{field['key']}"] == field["help"] for field in fields)


def main():
    settings()
    chunking()
    parsing()
    storage_and_ranking()
    limits_and_storage()
    secrets_and_policy()
    block_contract()
    print("[ok] Vector Store logic, storage and contract")


if __name__ == "__main__":
    main()
