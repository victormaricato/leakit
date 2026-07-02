"""leakit: continuation-free membership inference for closed language models.

Public API:
    LeakIt        - high-level scorer over any OpenAI-compatible endpoint
    ScoreResult   - per-document result
    self_concentration_word_jaccard / self_concentration_kgram - raw statistics
"""

from ._stats import self_concentration_kgram, self_concentration_word_jaccard
from .core import LeakIt, ScoreResult, percentile_of

__version__ = "0.1.0"

__all__ = [
    "LeakIt",
    "ScoreResult",
    "percentile_of",
    "self_concentration_word_jaccard",
    "self_concentration_kgram",
    "__version__",
]
