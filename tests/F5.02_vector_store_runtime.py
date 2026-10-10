"""FB4/FB5/FB6/FB7: real HTTP(S) to a loopback fake provider, then real Runs in both modes and three origins.

The embeddings server is the package-owned fixture on a dynamic loopback port. The
wallet is created in each disposable server; no real provider or credential is used.
"""

import ipaddress
import json
from pathlib import Path
import socket
import stat
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(Path(__file__).resolve().parent)]
from blocs.vector_store import logic, provider, store
from blocs.vector_store.block import VectorStoreBlock
from blocs.vector_store.logic import VectorError
from block_test_packages import install_test_package, surface_payload
from ui_smoke_common import (create_project_api, create_run_api, data_edge, display_node, graph_payload, graph_storage_dir,
                             http_json, isolated_server, stop_run_api, text_node, wait_for_run_predicate, wait_for_run_terminal)
from embeddings_fixture import embeddings_fixture, vector

TOKEN = "fixture-embeddings-key-4Tz"
REF = "secret://workspace/embeddings_fixture"
DOCS = [{"id": "cats", "text": "Cats purr and sleep on the warm sofa", "metadata": {"lang": "en"}},
        {"id": "dogs", "text": "Dogs bark at the mail carrier every morning", "metadata": {"lang": "en"}},
        {"id": "chats", "text": "Les chats dorment sur le canapé", "metadata": {"lang": "fr"}}]


def refused(call, code):
    try:
        call()
    except VectorError as exc:
        assert exc.code == code, (code, exc.code, str(exc))
        return exc
    raise AssertionError("Expected refusal " + code)


def protocol():
    """The HTTP client against the real loopback fixture: shape, auth, errors, limits, Stop, TLS and DNS policy."""
    defaults = VectorStoreBlock().default_config()
    with embeddings_fixture(token=TOKEN) as api:
        config = logic.configuration({**defaults, "base_url": api.base_url, "model": "fixture-embed",
                                      "api_key_ref": REF, "batch_size": 2})
        texts = ["alpha beta", "beta gamma", "gamma delta", "delta", "epsilon"]
        # - FB6 - The Bearer key is sent to the configured loopback endpoint; batches keep index order.
        assert provider.embed(texts, config, token=TOKEN) == [vector(text) for text in texts]
        assert len(api.calls) == 3 and all(call["auth_ok"] and call["path"] == "/v1/embeddings" for call in api.calls)
        assert [len(call["inputs"]) for call in api.calls] == [2, 2, 1] and api.calls[0]["user_agent"].startswith("BloxSmith-Vector-Store/")
        missing = refused(lambda: provider.embed(["x"], config, token=""), "auth")
        assert missing.details == {"http_status": 401, "provider_message": "Invalid API key provided."}
        refused(lambda: provider.embed(["x"], config, token="wrong-key-123"), "auth")
        for model, code, retryable in (("fixture-unavailable", "provider_unavailable", True), ("fixture-limited", "rate_limited", True),
                                       ("fixture-missing", "provider_rejected", False), ("fixture-bad-shape", "provider_response", False),
                                       ("fixture-base64", "provider_response", False), ("fixture-nan", "provider_response", False),
                                       ("fixture-gzip", "provider_response", False), ("fixture-zero", "provider_response", False)):
            other = {**config, "model": model}
            if model == "fixture-zero":
                refused(lambda: store.pack(provider.embed(["x"], other, token=TOKEN)[0]), code)
                continue
            assert refused(lambda: provider.embed(["x"], other, token=TOKEN), code).retryable is retryable
        assert "not found" in refused(lambda: provider.embed(["x"], {**config, "model": "fixture-missing"}, token=TOKEN), "provider_rejected").details["provider_message"]
        echoed = refused(lambda: provider.embed(["x"], {**config, "model": "fixture-echo"}, token=TOKEN), "provider_rejected")
        assert TOKEN not in json.dumps(echoed.envelope()) and "[redacted]" in echoed.details["provider_message"]
        large = {**config, "model": "fixture-large", "max_response_bytes": 65536}
        refused(lambda: provider.embed(["x", "y"], large, token=TOKEN), "provider_response")
        refused(lambda: provider.embed(["x"], {**config, "base_url": api.base_url + "/other"}, token=TOKEN), "provider_rejected")
        # - FB7 - Stop and the deadline abort a pending provider call promptly; nothing is waited for.
        stop = threading.Event()
        threading.Timer(0.5, stop.set).start()
        began = time.monotonic()
        refused(lambda: provider.embed(["x"], {**config, "model": "fixture-slow"}, token=TOKEN, cancel=stop.is_set), "cancelled")
        assert time.monotonic() - began < 3
        began = time.monotonic()
        refused(lambda: provider.embed(["x"], {**config, "model": "fixture-slow"}, token=TOKEN, deadline=time.monotonic() + 1), "timeout")
        assert time.monotonic() - began < 3
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = probe.getsockname()[1]
    assert refused(lambda: provider.embed(["x"], logic.configuration({**defaults, "base_url": f"http://127.0.0.1:{closed}/v1"})),
                   "network").retryable
    # - FB6 - HTTPS verifies the certificate chain and host name; a private CA is opt-in through ca_file.
    with embeddings_fixture(tls=True) as api:
        secure = logic.configuration({**defaults, "base_url": api.base_url, "model": "fixture-embed", "ca_file": api.ca_file})
        assert provider.embed(["tls text"], secure) == [vector("tls text")]
        named = {**secure, "base_url": f"https://localhost:{api.port}/v1"}
        assert provider.embed(["tls text"], named) == [vector("tls text")]
        refused(lambda: provider.embed(["tls text"], {**secure, "ca_file": ""}), "network")
    # - FB6 - DNS answers are vetted before any socket: no rebinding of localhost, metadata or link-local targets.
    private = str(ipaddress.ip_network(((192 << 24) | (168 << 16), 16))[7])
    answers = lambda address: [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 8000))]
    local = logic.configuration({**defaults, "base_url": "http://localhost:8000/v1", "api_key_ref": REF})
    remote = logic.configuration({**local, "allow_remote": True})
    insecure = logic.configuration({**remote, "allow_insecure_http": True})
    metadata = logic.configuration({**defaults, "base_url": "https://embed.example.test/v1", "allow_remote": True})
    for settings, address, message in ((local, private, "allow_remote"), (remote, private, "allow_insecure_http"),
                                       (insecure, private, "never sent"), (metadata, "169.254.169.254", "allow_remote")):
        with patch.object(provider.socket, "getaddrinfo", return_value=answers(address)), \
                patch.object(provider.socket, "socket", side_effect=AssertionError("socket opened")):
            assert message in str(refused(lambda: provider.embed(["x"], settings, token=TOKEN), "network"))
    print("[ok] Vector Store provider protocol", flush=True)


def document(node, message, port):
    return graph_payload("Vector Store", [
        text_node("source", "Message", message, 60, 160), node,
        display_node("matches", "Matches", 720, 40), display_node("result", "Result", 720, 220),
        display_node("error", "Error", 720, 400)], [
        data_edge("input", "source", 1, "vs", port), data_edge("to-matches", "vs", 1, "matches", 1),
        data_edge("to-result", "vs", 2, "result", 1), data_edge("to-error", "vs", 3, "error", 1)])


def start(server, project, node, message, port, mode):
    graph = document(node, message, port)
    (graph_storage_dir(server, project) / "graph.json").write_text(json.dumps({**graph, "graph_id": project["graph_id"]}), encoding="utf-8")
    http_json(server.base_url, f"/api/projects/{project['graph_id']}/graph/reload", method="POST", payload={})
    return create_run_api(server, graph, project_id=project["graph_id"], runtime_mode=mode)["run_id"]


def send(server, project, node, message, port, mode):
    """One Run per message; return the decoded documents the node published."""
    run = wait_for_run_terminal(server, start(server, project, node, message, port, mode), timeout_sec=60)
    assert run["status"] == "success", (mode, run.get("status"), run.get("logs"))
    assert TOKEN not in json.dumps(run), "The wallet key leaked into the Run state"
    values = run.get("output_values", {})
    return {name: json.loads(values[f"vs:{ident}"]["value"]) for name, ident in (("matches", 1), ("result", 2), ("error", 3))
            if values.get(f"vs:{ident}", {}).get("value")}


def stopped(server, project, node, mode):
    """Stop while the provider call is pending: the Run ends promptly and nothing is stored."""
    run_id = start(server, project, node, json.dumps(["a text that never gets a vector"]), 1, mode)
    wait_for_run_predicate(server, run_id, lambda state: state.get("node_statuses", {}).get("vs") == "running",
                           "The Vector Store node never started running.", timeout_sec=30)
    time.sleep(0.8)
    began = time.monotonic()
    stop_run_api(server, run_id)
    final = wait_for_run_predicate(server, run_id, lambda state: state.get("status") in {"cancelled", "failed", "success"},
                                   "The stopped Run did not end.", timeout_sec=30)
    assert time.monotonic() - began < 15, "Stop did not abort the pending provider call promptly"
    assert final["status"] == "cancelled", (mode, final.get("status"), final.get("logs"))
    values = final.get("output_values", {})
    assert not values.get("vs:1", {}).get("value") and not values.get("vs:2", {}).get("value"), values


def runs():
    block = VectorStoreBlock()
    with embeddings_fixture(token=TOKEN) as api:
        for mode in ("centralized", "zeromq_active"):
            for origin in (None, "managed", "linked"):
                with isolated_server() as server:
                    node = block.build_node_payload(node_id="vs", position={"x": 380, "y": 160})
                    config = {**block.default_config(), "base_url": api.base_url, "model": "fixture-embed", "api_key_ref": REF}
                    if origin:
                        model = install_test_package(server, "vector_store", origin=origin)
                        node["block_version"] = model["version"]
                        # - FB8 - Installed surfaces render without template residue, secrets or the endpoint on the card.
                        for surface in ("modal", "inspector_panel", "node_card", "mini_node_card"):
                            html = surface_payload(server, model, {**node, "config": config}, surface=surface)["html"]
                            assert "{{" not in html and TOKEN not in html
                            if surface in {"node_card", "mini_node_card"}:
                                assert "default · fixture-embed" in html and REF not in html and str(api.port) not in html
                            else:
                                assert 'data-block-config-field="base_url"' in html and 'data-owned-apply' in html
                    vs = {**node, "config": config}
                    project = create_project_api(server, document=document(vs, "seed", 1))["project"]
                    http_json(server.base_url, "/api/application/secrets/init", method="POST", payload={"password": "disposable-vector-fixture"})
                    http_json(server.base_url, "/api/application/secrets", method="POST", payload={"name": "embeddings_fixture", "value": TOKEN})
                    before = len(api.calls)
                    # - FB7 - Upsert in one Run, query in the next: the collection persists in node storage.
                    created = send(server, project, vs, json.dumps(DOCS), 1, mode)
                    assert set(created) == {"result"} and created["result"]["embedded"] == 3 and created["result"]["items"] == 3, created
                    assert api.calls[before]["auth_ok"] and len(api.calls[before]["inputs"]) == 3
                    found = send(server, project, vs, "Do cats purr on the sofa?", 2, mode)
                    assert set(found) == {"matches"} and found["matches"]["items"][0]["id"] == "cats", found
                    assert found["matches"]["context"].startswith("Cats purr") and found["matches"]["searched"] == 3
                    files = list((graph_storage_dir(server, project) / "instances" / "1" / "blocs" / "vs").glob("vectors-*.sqlite3"))
                    assert len(files) == 1 and stat.S_IMODE(files[0].stat().st_mode) == 0o600, files
                    if origin is None:
                        filtered = send(server, project, vs, json.dumps({"text": "cats", "filter": {"lang": "fr"}}), 2, mode)
                        assert [item["id"] for item in filtered["matches"]["items"]] == ["chats"], filtered
                        calls = len(api.calls)
                        stats = send(server, project, vs, json.dumps({"operation": "stats"}), 2, mode)
                        assert stats["result"]["items"] == 3 and stats["result"]["model"] == "fixture-embed" and len(api.calls) == calls
                        # - FB5 - Deletion stays refused until allow_delete is enabled.
                        denied = send(server, project, vs, json.dumps({"operation": "delete", "ids": ["dogs"]}), 1, mode)
                        assert set(denied) == {"error"} and denied["error"]["code"] == "delete_disabled", denied
                        allowed = {**vs, "config": {**config, "allow_delete": True}}
                        removed = send(server, project, allowed, json.dumps({"operation": "delete", "ids": ["dogs"]}), 1, mode)
                        assert removed["result"]["deleted"] == 1 and removed["result"]["items"] == 2, removed
                        # - FB4 - Another model never searches this collection; nothing reaches the provider.
                        other = {**vs, "config": {**config, "model": "fixture-other"}}
                        mismatch = send(server, project, other, "cats", 2, mode)
                        assert set(mismatch) == {"error"} and mismatch["error"]["code"] == "model_mismatch" and len(api.calls) == calls
                    # - FB6 - A locked wallet stops the call before any provider request.
                    http_json(server.base_url, "/api/application/secrets/lock", method="POST", payload={})
                    calls = len(api.calls)
                    locked = send(server, project, vs, json.dumps(["a brand new text"]), 1, mode)
                    assert set(locked) == {"error"} and locked["error"]["code"] == "secret" and len(api.calls) == calls, locked
                    if origin is None or (origin == "managed" and mode == "zeromq_active"):
                        http_json(server.base_url, "/api/application/secrets/unlock", method="POST",
                                  payload={"password": "disposable-vector-fixture"})
                        slow = {**vs, "config": {**config, "collection": "slow", "model": "fixture-slow", "timeout_sec": 120}}
                        stopped(server, project, slow, mode)
                        path = store.collection_path(graph_storage_dir(server, project) / "instances" / "1" / "blocs" / "vs", "slow")
                        assert not path.exists(), "A stopped upsert must not create the collection"
                print(f"[ok] Vector Store runtime {mode} {origin or 'bundled'}", flush=True)


def main():
    protocol()
    runs()


if __name__ == "__main__":
    main()
