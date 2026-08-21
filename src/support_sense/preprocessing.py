import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize

# Match common HTTP/HTTPS and www-style URLs.
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# Match one or more decimal digits.
NUMBER_PATTERN = re.compile(r"\d+")

# Match characters that are neither word characters nor whitespace.
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")

# Convert to a set because membership checks are the operation we need.
english_stop_words = set(stopwords.words("english"))


def validate_text(text: str) -> str:
    """Validate that preprocessing received text."""

    # Don't silently convert missing values into strings such as 'None'
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return text


def lowercase_text(text: str) -> str:
    """Convert text to lowercase."""

    return text.lower()


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim surrounding whitespace."""

    # split() handles spaces, tabs, and newlines.
    # Joining creates one consistent space between pieces.
    return " ".join(text.split())


def remove_urls(text: str) -> str:
    """Replace URLs with whitespace."""

    # Replace rather than concatenate surrounding words together.
    return URL_PATTERN.sub(" ", text)


def remove_numbers(text: str) -> str:
    """Replace sequences of digits with whitespace."""

    return NUMBER_PATTERN.sub(" ", text)


def remove_punctuation(text: str) -> str:
    """Replace punctuation characters with whitespace."""

    return PUNCTUATION_PATTERN.sub(" ", text)


def tokenize_text(text: str) -> list[str]:
    """Split text into tokens."""

    return wordpunct_tokenize(text)


def remove_stop_words(tokens: list[str], stop_words: set[str]) -> list[str]:
    """Remove configured stop words from a token sequence."""

    return [token for token in tokens if token not in stop_words]


def stem_tokens(tokens: list[str]) -> list[str]:

    stemmer = PorterStemmer()

    return [stemmer.stem(token) for token in tokens]


def lemmatize_tokens(tokens: list[str], pos: str = "n") -> list[str]:
    """Lemmatize every token with a configurable part of speech."""

    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(token, pos=pos) for token in tokens]


def preprocess_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_urls: bool = False,
    strip_numbers: bool = False,
    strip_punctuation: bool = False,
) -> str:
    """Apply configured preprocessing steps to text."""

    # Fail early for invalid input.
    text = validate_text(text)

    if lowercase:
        text = lowercase_text(text)

    # URLs should be handled before punctuation because URL punctuation
    # is part of what lets us recognize the URL pattern.
    if strip_urls:
        text = remove_urls(text)

    if strip_numbers:
        text = remove_numbers(text)

    if strip_punctuation:
        text = remove_punctuation(text)

    # Run this after replacements to collapse newly created spaces.
    text = normalize_whitespace(text)

    return text
