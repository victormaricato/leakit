"""Sampler behaviour with an injected fake client (no network)."""

import pytest

from leakit.sampler import Sampler, SamplerConfig, resolve_api_key


def _sampler(fake, **overrides):
    cfg = SamplerConfig(model="m", **overrides)
    return Sampler(cfg, api_key="k", client=fake)


def test_chat_collects_all_samples(fake_client):
    s = _sampler(fake_client(default=["x y z"]), n_samples=16, concurrency=4)
    out = s.sample("some prefix")
    assert len(out) == 16
    assert all(c == "x y z" for c in out)


def test_completion_mode(fake_client):
    s = _sampler(fake_client(default=["foo bar"]), n_samples=8, mode="completion", concurrency=2)
    out = s.sample("prefix")
    assert len(out) == 8
    assert out[0] == "foo bar"


def test_n_per_request_batches(fake_client):
    fc = fake_client(default=["a b", "c d"])
    s = _sampler(fc, n_samples=10, n_per_request=5, concurrency=4)
    out = s.sample("prefix")
    assert len(out) == 10
    # 10 samples / 5 per request = 2 calls
    assert fc.calls == 2


def test_retry_then_success(fake_client):
    fc = fake_client(default=["ok ok"], fail_times=2)
    s = _sampler(fc, n_samples=1, max_retries=5, concurrency=1)
    out = s.sample("prefix")
    assert out == ["ok ok"]
    assert fc.calls == 3  # 2 failures + 1 success


def test_dead_batch_is_skipped_not_fatal(fake_client):
    # Every call fails; with retries exhausted the run yields zero completions
    fc = fake_client(default=["never"], fail_times=99)
    s = _sampler(fc, n_samples=4, max_retries=2, concurrency=2)
    out = s.sample("prefix")
    assert out == []


def test_resolve_api_key_prefers_named(monkeypatch):
    monkeypatch.delenv("LEAKIT_API_KEY", raising=False)
    monkeypatch.setenv("MY_KEY", "secret")
    assert resolve_api_key("MY_KEY") == "secret"


def test_resolve_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("LEAKIT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        resolve_api_key()
