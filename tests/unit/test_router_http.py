import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from local_llm.paths import Paths
from local_llm.router import Router, RouterError
from local_llm.settings import Settings

from .fakes import FakeBackend, FakeHttp


def make(tmp_path, http=None, **settings):
    paths = Paths.from_env(env={}, home=tmp_path)
    return Router(paths, Settings(**settings), backend=FakeBackend(), http=http, binary="/x")


def test_api_methods_use_the_documented_endpoints(tmp_path):
    http = FakeHttp({("GET", "/health"): {"status": "ok"}})
    router = make(tmp_path, http=http)
    assert router.health() == {"status": "ok"}
    router.list_models()
    router.list_models(reload=True)
    router.load_model("m")
    router.unload_model("m")
    router.chat("m", "hi", max_tokens=3)
    assert http.calls == [
        ("GET", "/health", None),
        ("GET", "/models", None),
        ("GET", "/models?reload=1", None),
        ("POST", "/models/load", {"model": "m"}),
        ("POST", "/models/unload", {"model": "m"}),
        ("POST", "/v1/chat/completions",
         {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}),
    ]


class _Handler(BaseHTTPRequestHandler):
    seen: list[tuple[str, str, str | None, bytes]] = []

    def _reply(self, code, payload):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        _Handler.seen.append(("GET", self.path, self.headers.get("Authorization"), b""))
        if self.path == "/health":
            self._reply(200, b'{"status":"ok"}')
        elif self.path == "/plain":
            self._reply(200, b"not json")
        else:
            self._reply(404, b'{"error":{"code":404,"message":"no"}}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _Handler.seen.append(("POST", self.path, self.headers.get("Authorization"), body))
        self._reply(200, b'{"success":true}')

    def log_message(self, *args):  # keep test output quiet
        pass


@pytest.fixture
def server():
    _Handler.seen = []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()


def test_real_http_round_trip(tmp_path, server):
    router = make(tmp_path, port=server, api_key="k")
    assert router.health() == {"status": "ok"}
    assert router.load_model("m") == {"success": True}
    assert router.http("GET", "/missing", None) == {"error": {"code": 404, "message": "no"}}
    assert router.http("GET", "/plain", None) == {"raw": "not json"}
    get_health = _Handler.seen[0]
    post_load = _Handler.seen[1]
    assert get_health[2] == "Bearer k"
    assert json.loads(post_load[3]) == {"model": "m"}


def test_connection_refused_raises_router_error(tmp_path):
    router = make(tmp_path, port=1)  # nothing listens on port 1
    with pytest.raises(RouterError, match="Router is not answering"):
        router.health()
