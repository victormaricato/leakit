"""High-level scoring API: prefix extraction, sampling, self-concentration."""

from __future__ import annotations

from dataclasses import dataclass

from . import _stats
from .sampler import Sampler, SamplerConfig, resolve_api_key


@dataclass
class ScoreResult:
    document_id: str
    score: float
    n_requested: int
    n_returned: int
    statistic: str
    prefix: str
    completions: list[str]

    def as_dict(self, include_completions: bool = False) -> dict:
        d = {
            "document": self.document_id,
            "score": self.score,
            "statistic": self.statistic,
            "n_requested": self.n_requested,
            "n_returned": self.n_returned,
            "prefix_preview": self.prefix[:120],
        }
        if include_completions:
            d["completions"] = self.completions
        return d


def make_prefix(document: str, prefix_chars: int) -> str:
    """Take the conditioning prefix from the start of the document.

    prefix_chars == 0 means use the whole document. The prefix is what the model
    conditions on; the rest of the document is never sent (continuation-free).
    """
    text = document.strip()
    if prefix_chars and prefix_chars > 0:
        return text[:prefix_chars]
    return text


class LeakIt:
    """Continuation-free membership-inference scorer over an OpenAI-compatible API.

    Example
    -------
    >>> scorer = LeakIt(model="gpt-4o-mini")  # reads LEAKIT_API_KEY / OPENAI_API_KEY
    >>> result = scorer.score("In the beginning the Universe was created.")
    >>> result.score
    0.42
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        n_samples: int = 16,
        max_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        mode: str = "chat",
        concurrency: int = 8,
        n_per_request: int = 1,
        statistic: str = "word-jaccard",
        k: int = 5,
        prefix_chars: int = 256,
        client=None,
    ):
        if statistic not in _stats.STATISTICS:
            raise ValueError(
                f"unknown statistic {statistic!r}; choose from {sorted(_stats.STATISTICS)}"
            )
        self.statistic = statistic
        self.k = k
        self.prefix_chars = prefix_chars
        self.n_samples = n_samples
        key = api_key or resolve_api_key(api_key_env)
        self.sampler = Sampler(
            SamplerConfig(
                model=model,
                base_url=base_url,
                n_samples=n_samples,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                mode=mode,
                concurrency=concurrency,
                n_per_request=n_per_request,
            ),
            api_key=key,
            client=client,
        )

    def score(self, document: str, document_id: str = "<text>") -> ScoreResult:
        prefix = make_prefix(document, self.prefix_chars)
        completions = self.sampler.sample(prefix)
        value = _stats.compute(completions, self.statistic, self.k)
        return ScoreResult(
            document_id=document_id,
            score=value,
            n_requested=self.n_samples,
            n_returned=len(completions),
            statistic=self.statistic,
            prefix=prefix,
            completions=completions,
        )


def percentile_of(value: float, baseline: list[float]) -> float:
    """Fraction of baseline scores below ``value`` (0-100). Empty baseline -> nan."""
    if not baseline:
        return float("nan")
    below = sum(1 for b in baseline if b < value)
    return 100.0 * below / len(baseline)
