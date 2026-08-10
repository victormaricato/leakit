"""Extraction probe: masking, target location, and the attribution conjunction."""

import pytest

from leakit.extraction import (
    Extractor,
    find_identifiers,
    mask_identifier,
    prefix_before,
)

EMAIL = "alice@example.org"
DOC = f"Please direct all correspondence to the maintainer at {EMAIL} before filing."
OTHER = "An unrelated passage about glaciers, moraines, and outburst floods downstream."


def _extractor(client, **kw):
    kw.setdefault("n_samples", 3)
    kw.setdefault("prefix_chars", 32)
    return Extractor(model="m", api_key="k", client=client, **kw)


# ---- masking ----------------------------------------------------------------


def test_mask_email_keeps_only_shape():
    masked = mask_identifier(EMAIL)
    assert masked == "a***@e***.org"
    assert EMAIL not in masked


def test_mask_phone_hides_middle():
    assert mask_identifier("555-0142") == "55***42"


# ---- target location --------------------------------------------------------


def test_prefix_is_text_immediately_before_target():
    prefix = prefix_before(DOC, EMAIL, prefix_chars=20)
    assert DOC[DOC.find(EMAIL) - 20 : DOC.find(EMAIL)] == prefix
    assert EMAIL not in prefix, "the target must never be inside its own prefix"


def test_absent_target_raises_rather_than_measuring_nothing():
    with pytest.raises(ValueError, match="does not occur"):
        prefix_before(DOC, "bob@nowhere.test", prefix_chars=20)


def test_detects_identifiers_in_order_without_duplicates():
    text = f"{EMAIL} then {EMAIL} then carol@x.io"
    assert find_identifiers(text, "email") == [EMAIL, "carol@x.io"]


def test_unknown_detector_rejected():
    with pytest.raises(ValueError, match="unknown detector"):
        find_identifiers(DOC, "passport")


# ---- the attribution conjunction -------------------------------------------


def test_reproduced_and_control_clean_is_attributable(fake_client):
    """Leaks from the document's own context but not from unrelated context."""
    client = fake_client(
        script={"maintainer at": [f"{EMAIL} regards"], "downstream": ["no address"]},
        default=["no address"],
    )
    r = _extractor(client).probe(DOC, EMAIL, control_context=OTHER)
    assert r.reproduced and r.control_reproduced is False
    assert r.attributable is True


def test_globally_common_string_is_not_attributable(fake_client):
    """Emitted from any context, so the control discounts it."""
    client = fake_client(default=[f"contact {EMAIL}"])
    r = _extractor(client).probe(DOC, EMAIL, control_context=OTHER)
    assert r.reproduced and r.control_reproduced is True
    assert r.attributable is False


def test_no_reproduction_is_not_attributable(fake_client):
    client = fake_client(default=["nothing relevant here"])
    r = _extractor(client).probe(DOC, EMAIL, control_context=OTHER)
    assert r.reproduced is False
    assert r.attributable is False


def test_without_control_attribution_is_unknown_not_assumed(fake_client):
    """A probe with no control must not claim the leak is document-specific."""
    client = fake_client(script={"maintainer at": [f"{EMAIL} regards"]})
    r = _extractor(client).probe(DOC, EMAIL)
    assert r.reproduced is True
    assert r.control_reproduced is None
    assert r.attributable is None


def test_hits_counted_across_samples(fake_client):
    client = fake_client(
        script={"maintainer at": [f"{EMAIL} a", "nope", f"{EMAIL} b"]},
    )
    r = _extractor(client, n_samples=3).probe(DOC, EMAIL)
    assert r.n_hits == 2
    assert r.n_returned == 3


# ---- reporting hygiene ------------------------------------------------------


def test_output_masks_target_by_default(fake_client):
    client = fake_client(default=[f"{EMAIL}"])
    r = _extractor(client).probe(DOC, EMAIL, control_context=OTHER)
    assert r.as_dict()["target"] == "a***@e***.org"
    assert EMAIL not in str(r.as_dict())


def test_reveal_target_is_opt_in(fake_client):
    client = fake_client(default=[f"{EMAIL}"])
    r = _extractor(client).probe(DOC, EMAIL, reveal_target=True)
    assert r.as_dict()["target"] == EMAIL
