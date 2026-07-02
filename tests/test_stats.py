"""Statistic correctness, cross-checked against the paper's reference behaviour."""

import math

from leakit import _stats


def test_identical_completions_score_one():
    comps = ["the quick brown fox", "the quick brown fox", "the quick brown fox"]
    assert _stats.self_concentration_word_jaccard(comps) == 1.0
    assert _stats.self_concentration_kgram(comps, k=3) == 1.0


def test_disjoint_completions_score_zero():
    comps = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
    assert _stats.self_concentration_word_jaccard(comps) == 0.0


def test_word_jaccard_known_value():
    # {a,b} vs {b,c}: |∩|=1, |∪|=3 -> 1/3 ; single pair so mean == 1/3
    assert math.isclose(_stats.self_concentration_word_jaccard(["a b", "b c"]), 1 / 3)


def test_word_jaccard_mean_over_pairs():
    # pairs: (1,2)=1/3, (1,3)=0, (2,3)=0  -> mean = 1/9
    val = _stats.self_concentration_word_jaccard(["a b", "b c", "x y"])
    assert math.isclose(val, (1 / 3) / 3)


def test_fewer_than_two_is_zero():
    assert _stats.self_concentration_word_jaccard([]) == 0.0
    assert _stats.self_concentration_word_jaccard(["only one"]) == 0.0


def test_empty_strings_do_not_crash():
    assert _stats.self_concentration_word_jaccard(["", ""]) == 0.0
    assert _stats.self_concentration_kgram(["", "abc"], k=5) == 0.0


def test_member_scores_above_nonmember():
    member = [
        "born in 1809 in Hardin County",
        "born in 1809 in Hardin County, Kentucky",
        "born in 1809 in Hardin",
    ]
    nonmember = [
        "the weather today is",
        "I think that maybe",
        "purple monkey dishwasher",
    ]
    assert _stats.self_concentration_word_jaccard(
        member
    ) > _stats.self_concentration_word_jaccard(nonmember)


def test_dispatch_unknown_raises():
    try:
        _stats.compute(["a", "b"], statistic="nope")
    except ValueError as e:
        assert "unknown statistic" in str(e)
    else:
        raise AssertionError("expected ValueError")
