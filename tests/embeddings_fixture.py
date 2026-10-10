"""Owned loopback OpenAI-compatible embeddings server. No request leaves loopback; no real key is used.

Vectors are deterministic hashed bags of words: texts sharing words are closer, so
rankings can be recomputed exactly by the tests. Special model names select failures.
"""

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import ssl
import subprocess
import tempfile
import threading
from types import SimpleNamespace

DIMENSION = 64


def vector(text, dimension=DIMENSION):
    """Hashed bag of words plus a constant component, so no vector is ever zero."""
    values = [0.0] * dimension
    for word in re.findall(r"\w+", text.lower()):
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        values[digest[0] % (dimension - 1)] += 1.0 + digest[1] / 255.0
    values[dimension - 1] = 0.05
    return values


def fake_embedder(dimension=DIMENSION, calls=None):
    """In-process stand-in with the provider.embed signature, for storage/logic tests."""
    def embed(texts, config, *, deadline=None, cancel=lambda: False):
        if calls is not None:
            calls.append({"model": config["model"], "inputs": list(texts)})
        return [vector(text, dimension) for text in texts]
    return embed


@contextmanager
def embeddings_fixture(*, token=None, tls=False):
    """Serve POST /v1/embeddings on a dynamic loopback port and stop every owned thread."""
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        closing, calls = threading.Event(), []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def reply(self, code, payload, headers=None):
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
                if closing.is_set():
                    return
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Connection", "close")
                    for key, value in (headers or {}).items():
                        self.send_header(key, value)
                    self.end_headers()
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionError, ssl.SSLError):
                    pass
                finally:
                    self.close_connection = True

            def do_POST(self):
                body = self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 8388608))
                auth = self.headers.get("Authorization", "")
                try:
                    request = json.loads(body)
                except ValueError:
                    request = {}
                model, inputs = request.get("model"), request.get("input")
                calls.append({"path": self.path, "model": model, "inputs": inputs, "auth_present": bool(auth),
                              "auth_ok": bool(token) and auth == "Bearer " + token, "user_agent": self.headers.get("User-Agent", "")})
                if self.path != "/v1/embeddings":
                    return self.reply(404, {"error": {"message": "unknown route"}})
                if token and auth != "Bearer " + token:
                    return self.reply(401, {"error": {"message": "Invalid API key provided."}})
                if not isinstance(inputs, list) or not all(isinstance(item, str) for item in inputs):
                    return self.reply(400, {"error": {"message": "input must be an array of strings"}})
                if model == "fixture-slow":
                    closing.wait(30)
                    return self.reply(503, {"error": {"message": "slow fixture released"}})
                if model == "fixture-unavailable":
                    return self.reply(503, {"error": {"message": "model is loading, retry later"}})
                if model == "fixture-limited":
                    return self.reply(429, {"error": {"message": "rate limit reached"}})
                if model == "fixture-missing":
                    return self.reply(404, {"error": {"message": "model 'fixture-missing' not found"}})
                if model == "fixture-echo":
                    return self.reply(400, {"error": {"message": "rejected header " + auth}})
                if model == "fixture-bad-shape":
                    return self.reply(200, {"object": "list"})
                if model == "fixture-base64":
                    return self.reply(200, {"data": [{"index": i, "embedding": "AAAA"} for i, _ in enumerate(inputs)]})
                if model == "fixture-nan":
                    return self.reply(200, b'{"data":[{"index":0,"embedding":[NaN,1.0]}]}')
                if model == "fixture-gzip":
                    return self.reply(200, {"data": []}, {"Content-Encoding": "gzip"})
                if model == "fixture-large":
                    return self.reply(200, {"data": [{"index": i, "embedding": [0.5] * 8192} for i, _ in enumerate(inputs)]})
                dimension = 6 if model == "fixture-small" else DIMENSION
                values = [vector(text, dimension) for text in inputs]
                if model == "fixture-zero":
                    values = [[0.0] * dimension for _ in inputs]
                data = [{"object": "embedding", "index": index, "embedding": item} for index, item in enumerate(values)]
                data.reverse()  # Clients must order by index, not by position.
                return self.reply(200, {"object": "list", "model": model, "data": data,
                                        "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)}})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        ca = ""
        if tls:
            assert shutil.which("openssl"), "OpenSSL is required for the disposable HTTPS fixture"
            ca, key = directory / "ca.pem", directory / "key.pem"
            subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                            "-keyout", str(key), "-out", str(ca), "-subj", "/CN=localhost",
                            "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1"], check=True, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(ca, key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
            ca = str(ca)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            scheme = "https" if tls else "http"
            yield SimpleNamespace(base_url=f"{scheme}://127.0.0.1:{server.server_port}/v1", ca_file=ca, calls=calls,
                                  port=server.server_port)
        finally:
            closing.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            assert not thread.is_alive()
