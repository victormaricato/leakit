"""leakit: black-box training-data extraction and leakage probing for language models.

Two measurements, both needing nothing but a sampling endpoint:

    Extractor     - per-document verbatim extraction: does the model reproduce a
                    specific string from the context preceding it, and not from a
                    mismatched context? This is the paper's positive result.
    LeakIt        - aggregate sampling statistics (self-concentration). Reported
                    for completeness; the paper shows these do NOT beat a
                    model-free blind baseline as aggregate membership classifiers.
"""

from ._stats import self_concentration_kgram, self_concentration_word_jaccard
from .core import LeakIt, ScoreResult, percentile_of
from .extraction import (
    ExtractionResult,
    Extractor,
    find_identifiers,
    mask_identifier,
    prefix_before,
)

__version__ = "0.2.0"

__all__ = [
    "Extractor",
    "ExtractionResult",
    "find_identifiers",
    "mask_identifier",
    "prefix_before",
    "LeakIt",
    "ScoreResult",
    "percentile_of",
    "self_concentration_word_jaccard",
    "self_concentration_kgram",
    "__version__",
]
