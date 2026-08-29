"""Deterministic text comparison metrics.

None of these call a language model. They are fast, free, reproducible,
and for a lot of questions they are simply the right tool -- an exact
match check does not need a judge's opinion.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter

from ragrank.bridge.pydantic import Field
from ragrank.dataset import DataNode
from ragrank.metric.base import DeterministicMetric, MetricType

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Normalise text for comparison.

    Lowercases, strips punctuation and articles, and collapses runs of
    whitespace -- the standard SQuAD normalisation, so that "The Paris."
    and "paris" compare equal.

    Args:
        text (str): The text to normalise.

    Returns:
        str: The normalised text.
    """
    text = text.casefold().translate(_PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split normalised text into tokens.

    Args:
        text (str): The text to tokenise.

    Returns:
        list[str]: The tokens.
    """
    normalised = normalize(text)
    return normalised.split() if normalised else []


def levenshtein(left: str, right: str) -> int:
    """Edit distance between two strings.

    Iterative two-row implementation: O(len(left) * len(right)) time and
    O(min) space, no dependency needed.

    Args:
        left (str): The first string.
        right (str): The second string.

    Returns:
        int: The number of single character edits between them.
    """
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def lcs_length(left: list[str], right: list[str]) -> int:
    """Length of the longest common subsequence of two token lists.

    Args:
        left (list[str]): The first token list.
        right (list[str]): The second token list.

    Returns:
        int: The length of the longest common subsequence.
    """
    if not left or not right:
        return 0

    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for j, right_token in enumerate(right, start=1):
            current.append(
                previous[j - 1] + 1
                if left_token == right_token
                else max(previous[j], current[j - 1])
            )
        previous = current
    return previous[-1]


class ReferenceMetric(DeterministicMetric):
    """Base for metrics that compare the response to a reference."""

    @property
    def required_columns(self) -> set[str]:
        """Fields this metric needs.

        Returns:
            set[str]: `response` and `reference`.
        """
        return {"response", "reference"}


class ExactMatch(ReferenceMetric):
    """1.0 when the response matches the reference after normalisation."""

    metric_type: MetricType = Field(default=MetricType.BINARY)

    @property
    def name(self) -> str:
        """The metric's name.

        Returns:
            str: The name.
        """
        return "Exact Match"

    def compute(self, data: DataNode) -> float | None:
        """Compare response and reference.

        Args:
            data (DataNode): The row to score.

        Returns:
            float | None: 1.0 or 0.0, or None without a reference.
        """
        if data.reference is None:
            return None
        return float(
            normalize(data.response) == normalize(data.reference)
        )


class StringPresence(ReferenceMetric):
    """1.0 when the reference appears somewhere in the response."""

    metric_type: MetricType = Field(default=MetricType.BINARY)

    @property
    def name(self) -> str:
        """The metric's name.

        Returns:
            str: The name.
        """
        return "String Presence"

    def compute(self, data: DataNode) -> float | None:
        """Check for the reference inside the response.

        Args:
            data (DataNode): The row to score.

        Returns:
            float | None: 1.0 or 0.0, or None without a reference.
        """
        if data.reference is None:
            return None
        return float(
            normalize(data.reference) in normalize(data.response)
        )


class LevenshteinRatio(ReferenceMetric):
    """Character level similarity to the reference, in [0, 1]."""

    @property
    def name(self) -> str:
        """The metric's name.

        Returns:
            str: The name.
        """
        return "Levenshtein Ratio"

    def compute(self, data: DataNode) -> float | None:
        """Compute 1 - normalised edit distance.

        Args:
            data (DataNode): The row to score.

        Returns:
            float | None: The similarity, or None without a reference.
        """
        if data.reference is None:
            return None

        left, right = (
            normalize(data.response),
            normalize(data.reference),
        )
        longest = max(len(left), len(right))
        if longest == 0:
            return 1.0
        return 1.0 - levenshtein(left, right) / longest


class TokenF1(ReferenceMetric):
    """Token overlap F1 against the reference.

    The standard SQuAD-style measure: forgiving about word order and
    phrasing, unforgiving about missing or invented content.
    """

    @property
    def name(self) -> str:
        """The metric's name.

        Returns:
            str: The name.
        """
        return "Token F1"

    def compute(self, data: DataNode) -> float | None:
        """Compute token level F1.

        Args:
            data (DataNode): The row to score.

        Returns:
            float | None: The F1, or None without a reference.
        """
        if data.reference is None:
            return None

        predicted = Counter(tokenize(data.response))
        expected = Counter(tokenize(data.reference))
        if not predicted or not expected:
            return float(not predicted and not expected)

        overlap = sum((predicted & expected).values())
        if overlap == 0:
            return 0.0

        precision = overlap / sum(predicted.values())
        recall = overlap / sum(expected.values())
        return 2 * precision * recall / (precision + recall)


class RougeL(ReferenceMetric):
    """ROUGE-L F1 -- longest common subsequence against the reference."""

    @property
    def name(self) -> str:
        """The metric's name.

        Returns:
            str: The name.
        """
        return "ROUGE-L"

    def compute(self, data: DataNode) -> float | None:
        """Compute the LCS based F1.

        Args:
            data (DataNode): The row to score.

        Returns:
            float | None: The score, or None without a reference.
        """
        if data.reference is None:
            return None

        predicted = tokenize(data.response)
        expected = tokenize(data.reference)
        if not predicted or not expected:
            return float(not predicted and not expected)

        common = lcs_length(predicted, expected)
        if common == 0:
            return 0.0

        precision = common / len(predicted)
        recall = common / len(expected)
        return 2 * precision * recall / (precision + recall)


class JsonValid(DeterministicMetric):
    """1.0 when the response parses as JSON."""

    metric_type: MetricType = Field(default=MetricType.BINARY)

    @property
    def name(self) -> str:
        """The metric's name.

        Returns:
            str: The name.
        """
        return "JSON Valid"

    @property
    def required_columns(self) -> set[str]:
        """Fields this metric needs.

        Returns:
            set[str]: Just `response`.
        """
        return {"response"}

    def compute(self, data: DataNode) -> float | None:
        """Try to parse the response as JSON.

        Args:
            data (DataNode): The row to score.

        Returns:
            float | None: 1.0 if it parses, else 0.0.
        """
        try:
            json.loads(data.response)
        except (ValueError, TypeError):
            return 0.0
        return 1.0


exact_match = ExactMatch()
string_presence = StringPresence()
levenshtein_ratio = LevenshteinRatio()
token_f1 = TokenF1()
rouge_l = RougeL()
json_valid = JsonValid()
