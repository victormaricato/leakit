"""Sampling backend built on the OpenAI Python SDK.

Any OpenAI-compatible endpoint works: set ``base_url`` to the provider you are
probing (OpenAI, Anthropic via its OpenAI-compatible route, OpenRouter, vLLM,
Together, a local server, ...) and supply the matching API key. The attack only
needs a sampling endpoint; it never reads logits.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

# The continuation instruction used in chat mode. Closed chat APIs do not expose
# a raw text-completion surface, so we ask the model to continue the passage
# verbatim. Base/text models should use mode="completion" for the unbiased
# sampling distribution the paper studies.
_CONTINUE_SYSTEM = (
    "You continue text. Given the beginning of a passage, write the text that "
    "most plausibly comes next. Output only the continuation, with no preamble, "
    "quotation marks, or commentary."
)

DEFAULT_API_KEY_ENVS = ("LEAKIT_API_KEY", "OPENAI_API_KEY")


def resolve_api_key(api_key_env: str | None = None) -> str:
    """Resolve the API key from the environment.

    If ``api_key_env`` is given, only that variable is consulted. Otherwise the
    default chain (LEAKIT_API_KEY then OPENAI_API_KEY) is tried.
    """
    candidates = (api_key_env,) if api_key_env else DEFAULT_API_KEY_ENVS
    for name in candidates:
        if name and os.environ.get(name):
            return os.environ[name]
    tried = ", ".join(c for c in candidates if c)
    raise RuntimeError(
        f"no API key found in environment (looked at: {tried}). "
        f"Export your provider key, e.g. `export {candidates[0]}=sk-...`."
    )


@dataclass
class SamplerConfig:
    model: str
    base_url: str | None = None
    n_samples: int = 16
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    mode: str = "chat"  # "chat" or "completion"
    concurrency: int = 8
    n_per_request: int = 1  # set >1 to batch via the provider's `n` param
    timeout: float = 120.0
    max_retries: int = 4
    extra_body: dict = field(default_factory=dict)


class Sampler:
    """Draws continuations of a prefix from an OpenAI-compatible endpoint."""

    def __init__(self, config: SamplerConfig, api_key: str, client=None):
        self.config = config
        # `client` is injectable for testing; defaults to a real OpenAI client.
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=0,  # we handle retries/backoff ourselves
            )
        self.client = client

    def _request(self, prefix: str, n: int) -> list[str]:
        """One API call returning up to ``n`` continuations."""
        cfg = self.config
        if cfg.mode == "completion":
            resp = self.client.completions.create(
                model=cfg.model,
                prompt=prefix,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                n=n,
                extra_body=cfg.extra_body or None,
            )
            return [choice.text or "" for choice in resp.choices]
        elif cfg.mode == "chat":
            resp = self.client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": _CONTINUE_SYSTEM},
                    {"role": "user", "content": prefix},
                ],
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                n=n,
                extra_body=cfg.extra_body or None,
            )
            return [(choice.message.content or "") for choice in resp.choices]
        raise ValueError(f"unknown mode {cfg.mode!r}; use 'chat' or 'completion'")

    def _request_with_retry(self, prefix: str, n: int) -> list[str]:
        import time

        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                return self._request(prefix, n)
            except Exception as exc:  # noqa: BLE001 - provider errors are heterogeneous
                last_exc = exc
                if not _is_retryable(exc) or attempt == self.config.max_retries - 1:
                    break
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"sampling request failed: {last_exc}") from last_exc

    def sample(self, prefix: str) -> list[str]:
        """Return up to ``n_samples`` continuations of ``prefix``.

        Requests are batched by ``n_per_request`` and run concurrently. Failed
        requests are skipped; the returned list may be shorter than n_samples.
        """
        cfg = self.config
        per = max(1, cfg.n_per_request)
        # Build a list of per-request batch sizes summing to n_samples.
        batches: list[int] = []
        remaining = cfg.n_samples
        while remaining > 0:
            batches.append(min(per, remaining))
            remaining -= batches[-1]

        completions: list[str] = []
        workers = max(1, min(cfg.concurrency, len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self._request_with_retry, prefix, b) for b in batches]
            for fut in as_completed(futures):
                try:
                    completions.extend(fut.result())
                except Exception:  # noqa: BLE001 - one dead batch shouldn't kill the run
                    continue
        return completions


def _is_retryable(exc: Exception) -> bool:
    """Heuristic: retry on rate limits, timeouts, and 5xx; not on 4xx/auth."""
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    name = type(exc).__name__.lower()
    return any(tok in name for tok in ("timeout", "connection", "ratelimit", "apierror"))
