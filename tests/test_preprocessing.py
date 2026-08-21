import pytest

from support_sense.preprocessing import (
    lemmatize_tokens,
    lowercase_text,
    normalize_whitespace,
    preprocess_text,
    remove_stop_words,
    stem_tokens,
)


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("HELLO", "hello"),
        ("Payment FAILED", "payment failed"),
        ("already lowercase", "already lowercase"),
    ],
)
def test_lowercase_text_converts_uppercase(raw_text: str, expected: str):
    # Uppercase variants should be normalized.
    assert lowercase_text(raw_text) == expected


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("payment     failed", "payment failed"),
    ],
)
def test_normalize_whitespace_collapses_extra_spaces(raw_text: str, expected: str):
    # Multiple spaces should become one.
    assert normalize_whitespace(raw_text) == expected


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("payment\tfailed\nagain", "payment failed again"),
    ],
)
def test_normalize_whitespace_handles_tabs_and_newlines(raw_text: str, expected: str):
    # Different whitespace types should be normalized consistently.
    assert normalize_whitespace(raw_text) == expected


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("Payment failed!!! Please help.", "payment failed please help"),
    ],
)
def test_preprocess_text_can_remove_punctuation(raw_text: str, expected: str):
    # Removing punctuation should still preserve word boundaries.
    result = preprocess_text(
        raw_text,
        strip_punctuation=True,
    )

    assert result == expected


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("Refund pending for 5 days", "refund pending for days"),
    ],
)
def test_preprocess_text_can_remove_numbers(raw_text: str, expected: str):
    # Number removal should be optional.
    result = preprocess_text(
        raw_text,
        strip_numbers=True,
    )

    assert result == expected


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("Error 404", "error 404"),
    ],
)
def test_preprocess_text_preserves_numbers_by_default(raw_text: str, expected: str):
    # Conservative preprocessing should preserve numbers.
    result = preprocess_text(raw_text)

    assert result == expected


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("", ""),
    ],
)
def test_preprocess_text_handles_empty_string(raw_text: str, expected: str):
    # An empty but valid string should remain empty.
    assert preprocess_text(raw_text) == expected


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("    \n\t ", ""),
    ],
)
def test_preprocess_text_handles_whitespace_only_string(raw_text: str, expected: str):
    # Whitespace-only input becomes an empty normalized string.
    assert preprocess_text(raw_text) == expected


def test_preprocess_text_rejects_none():
    # Missing text should not silently become the string "None".
    with pytest.raises(TypeError):
        preprocess_text(None)  # type: ignore[arg-type]


def test_remove_stop_words_uses_provided_word_set():
    # Use a small deterministic stop-word set for the unit test.
    tokens = ["my", "payment", "has", "failed"]
    stop_words = {"my", "has"}

    result = remove_stop_words(tokens, stop_words)

    assert result == ["payment", "failed"]


def test_stem_tokens_returns_one_stem_per_token():
    # Stemming should preserve the number and order of tokens.
    tokens = ["connected", "connecting"]

    result = stem_tokens(tokens)

    assert len(result) == len(tokens)


def test_lemmatize_tokens_reduces_plural_nouns():
    # WordNet's noun lemmatization maps this plural to its lemma.
    result = lemmatize_tokens(["dogs"])

    assert result == ["dog"]
