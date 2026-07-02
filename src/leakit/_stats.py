"""Self-concentration statistics for continuation-free membership inference.

These are the membership signals defined in the paper. The statistic operates
purely on a set of sampled continuations: it never sees a gold continuation and
needs no model internals. Higher values indicate a more concentrated sampling
distribution, which the paper shows is predictive of training-set membership.

Pure Python, no third-party dependencies, so the package stays lightweight.
"""

from __future__ import annotations

from itertools import combinations


def _word_set(text: str) -> set[str]:
    return set(text.split())


def _kgram_set(text: str, k: int) -> set[str]:
    if len(text) < k:
        return set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _mean_pairwise_jaccard(sets: list[set[str]]) -> float:
    if len(sets) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for a, b in combinations(sets, 2):
        union = a | b
        if not union:
            continue
        total += len(a & b) / len(union)
        pairs += 1
    return total / pairs if pairs else 0.0


def self_concentration_word_jaccard(completions: list[str]) -> float:
    """Parameter-free self-concentration: mean pairwise word-set Jaccard.

    Each completion is reduced to its set of whitespace-delimited tokens, and we
    average the Jaccard similarity across all unordered pairs. This is the
    headline statistic in the paper (no n-gram size to tune).
    """
    return _mean_pairwise_jaccard([_word_set(c) for c in completions])


def self_concentration_kgram(completions: list[str], k: int = 5) -> float:
    """Self-concentration over character k-grams: mean pairwise k-gram Jaccard."""
    return _mean_pairwise_jaccard([_kgram_set(c, k) for c in completions])


STATISTICS = {
    "word-jaccard": lambda completions, k: self_concentration_word_jaccard(completions),
    "kgram": self_concentration_kgram,
}


def compute(
    completions: list[str], statistic: str = "word-jaccard", k: int = 5
) -> float:
    """Dispatch to the named statistic. Raises ValueError on unknown names."""
    try:
        fn = STATISTICS[statistic]
    except KeyError:
        raise ValueError(
            f"unknown statistic {statistic!r}; choose from {sorted(STATISTICS)}"
        ) from None
    return fn(completions, k)
