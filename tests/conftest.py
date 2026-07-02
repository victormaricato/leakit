"""Shared fakes that mimic the openai client response shape."""

import threading
import types

import pytest


def _chat_response(texts):
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(message=types.SimpleNamespace(content=t)) for t in texts
        ]
    )


def _completion_response(texts):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(text=t) for t in texts]
    )


class FakeAPIError(Exception):
    """Mimics a retryable provider error (carries an HTTP status code)."""

    def __init__(self, status_code=503):
        super().__init__(f"fake {status_code}")
        self.status_code = status_code


class FakeClient:
    """Returns canned continuations keyed by a substring of the prefix.

    Successive requests rotate through the pool (so independent single-sample
    calls vary, like a real model). `fail_times` raises a retryable error on the
    first N calls to exercise backoff.
    """

    def __init__(self, script=None, default=None, fail_times=0):
        self.script = script or {}
        self.default = default or ["a b c", "a b d", "a b e"]
        self.fail_times = fail_times
        self.calls = 0
        self._offset = 0
        self._lock = threading.Lock()
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._chat))
        self.completions = types.SimpleNamespace(create=self._completion)

    def _pick(self, prompt):
        for key, comps in self.script.items():
            if key in prompt:
                return comps
        return self.default

    def _emit(self, texts, n):
        with self._lock:
            start = self._offset
            self._offset += n
        return [texts[(start + i) % len(texts)] for i in range(n)]

    def _guard(self):
        with self._lock:
            self.calls += 1
            failed = self.calls <= self.fail_times
        if failed:
            raise FakeAPIError(503)

    def _chat(self, *, model, messages, n=1, **kw):
        self._guard()
        prompt = messages[-1]["content"]
        return _chat_response(self._emit(self._pick(prompt), n))

    def _completion(self, *, model, prompt, n=1, **kw):
        self._guard()
        return _completion_response(self._emit(self._pick(prompt), n))


@pytest.fixture
def fake_client():
    return FakeClient
