"""Live HTTP integration: real openai SDK against a local OpenAI-shaped server.

Proves the actual network path (request shape, base_url routing, response
parsing) without any paid API. The server returns concentrated continuations
for a "MEMBER" prefix and diffuse ones for a "NOVEL" prefix.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from leakit import LeakIt

# Members: identical continuations (model concentrates on memorised text).
# Novel: four distinct continuations rotated across requests (diffuse distribution).
MEMBER_CONTINUATIONS = ["in Hardin County Kentucky"]
NOVEL_CONTINUATIONS = ["a winding road ahead", "thoughts about nothing much",
                       "quux frobnicate widget", "the seventeenth of never"]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _emit(self, pool, n):
        # Rotate a server-wide cursor so independent requests vary, like a model.
        srv = self.server
        with srv._lock:  # type: ignore[attr-defined]
            start = srv._cursor  # type: ignore[attr-defined]
            srv._cursor += n  # type: ignore[attr-defined]
        return [pool[(start + i) % len(pool)] for i in range(n)]

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        n = int(body.get("n", 1) or 1)

        if self.path.endswith("/chat/completions"):
            prompt = body["messages"][-1]["content"]
            pool = MEMBER_CONTINUATIONS if "MEMBER" in prompt else NOVEL_CONTINUATIONS
            texts = self._emit(pool, n)
            choices = [{"index": i, "message": {"role": "assistant",
                        "content": t}, "finish_reason": "stop"}
                       for i, t in enumerate(texts)]
        else:  # /completions
            prompt = body.get("prompt", "")
            pool = MEMBER_CONTINUATIONS if "MEMBER" in prompt else NOVEL_CONTINUATIONS
            texts = self._emit(pool, n)
            choices = [{"index": i, "text": t, "finish_reason": "stop"}
                       for i, t in enumerate(texts)]

        payload = json.dumps({
            "id": "cmpl-test", "object": "chat.completion", "model": body.get("model", "m"),
            "choices": choices,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def local_openai_server():
    import threading as _t
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server._cursor = 0
    server._lock = _t.Lock()
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


def test_live_http_chat_mode_member_vs_novel(local_openai_server):
    scorer = LeakIt(model="local-test", base_url=local_openai_server, api_key="test-key",
                    n_samples=8, concurrency=4, mode="chat")
    member = scorer.score("MEMBER: the sixteenth president was born", document_id="member")
    novel = scorer.score("NOVEL: my grocery list for tuesday", document_id="novel")
    assert member.n_returned == 8
    assert novel.n_returned == 8
    assert member.score == 1.0          # identical continuations -> full concentration
    assert member.score > novel.score


def test_live_http_completion_mode(local_openai_server):
    scorer = LeakIt(model="local-test", base_url=local_openai_server, api_key="test-key",
                    n_samples=4, concurrency=2, mode="completion")
    res = scorer.score("MEMBER prefix", document_id="m")
    assert res.n_returned == 4
    assert res.score == 1.0


def test_live_http_n_per_request_batch(local_openai_server):
    scorer = LeakIt(model="local-test", base_url=local_openai_server, api_key="test-key",
                    n_samples=12, n_per_request=4, concurrency=3, mode="chat")
    res = scorer.score("MEMBER prefix", document_id="m")
    assert res.n_returned == 12
