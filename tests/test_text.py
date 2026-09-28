"""Spec strings must tokenize to the same terms however the supplier glued them together."""
import pytest

from pmatch.text import tokenize


@pytest.mark.parametrize("text, expected", [
    ("DC12V", ["dc", "12v"]),
    ("12V", ["12v"]),
    ("12V1.5A", ["12v", "1.5a"]),
    ("AC100~240V", ["ac", "100", "240v"]),
    ("24VDC", ["24v", "dc"]),
    ("Brushed DC", ["brushed", "dc"]),
    ("IP65", ["ip65"]),
    ("2600mAh", ["2600mah"]),
])
def test_spec_tokens(text, expected):
    assert tokenize(text) == expected
