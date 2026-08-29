"""Parsing an LLM's answer into a score.

Language models are not reliable float emitters. They add prose, they add
units, they refuse, they hedge. This module turns whatever came back into
either a number or an honest `None` -- it never raises.
"""

from __future__ import annotations

import json
import re
from typing import NamedTuple

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_FENCE = re.compile(r"^\s*```(?:\w+)?\s*|\s*```\s*$")


class ParsedScore(NamedTuple):
    """The outcome of parsing one LLM response.

    Attributes:
        score (float | None): The parsed score, or None if unparseable.
        error (str | None): Why parsing failed, when it did.
    """

    score: float | None
    error: str | None


def parse_score(
    text: str,
    *,
    score_range: tuple[float, float] = (0.0, 1.0),
    rubric: dict[str, float] | None = None,
) -> ParsedScore:
    """Extract a score from an LLM response.

    When a `rubric` is given the response is treated as a choice: the model
    is expected to emit one of the rubric's keys, and the numeric scale
    lives here rather than in the model. This is far more reliable than
    asking for a float, so prefer it.

    Without a rubric the response is read as a number, and anything outside
    `score_range` is rejected rather than silently returned.

    Args:
        text (str): The raw LLM response.
        score_range (tuple[float, float]): Inclusive bounds for the score.
        rubric (dict[str, float] | None): Choice label to score mapping.

    Returns:
        ParsedScore: The score, or None with an explanation.
    """
    if text is None:
        return ParsedScore(None, "the LLM returned no content")

    cleaned = _FENCE.sub("", text).strip()
    if not cleaned:
        return ParsedScore(
            None, "the LLM returned an empty response"
        )

    if rubric:
        return _parse_choice(cleaned, rubric)
    return _parse_number(cleaned, score_range)


def _parse_choice(
    text: str, rubric: dict[str, float]
) -> ParsedScore:
    """Match the response against a rubric's choice labels."""
    stripped = text.strip().strip(".:()[]\"' ")

    for label, value in rubric.items():
        if stripped.casefold() == label.casefold():
            return ParsedScore(value, None)

    # The label may be embedded, e.g. "Answer: B" or "B) because ...".
    for label, value in rubric.items():
        pattern = rf"(?<![\w-]){re.escape(label)}(?![\w-])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return ParsedScore(value, None)

    choices = ", ".join(rubric)
    return ParsedScore(
        None,
        f"expected one of [{choices}], got {text[:80]!r}",
    )


def _parse_number(
    text: str, score_range: tuple[float, float]
) -> ParsedScore:
    """Read a number out of the response and bounds-check it."""
    low, high = score_range

    match = _NUMBER.search(text)
    if match is None:
        return ParsedScore(None, f"no number found in {text[:80]!r}")

    value = float(match.group())
    if not low <= value <= high:
        return ParsedScore(
            None,
            f"score {value} is outside the valid range [{low}, {high}]",
        )
    return ParsedScore(value, None)


_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


def parse_list(text: str) -> list[str]:
    """Extract a list of statements from an LLM response.

    Models asked for a list answer with a JSON array, a numbered list,
    a bulleted list, or bare lines, depending on the model and the day.
    All four are accepted; anything unusable yields an empty list
    rather than an exception.

    Args:
        text (str): The raw LLM response.

    Returns:
        list[str]: The extracted items, in order, without duplicates.
    """
    if not text:
        return []

    cleaned = _FENCE.sub("", text).strip()
    if not cleaned:
        return []

    items = _from_json(cleaned)
    if items is None:
        items = [
            _BULLET.sub("", line).strip().strip('"')
            for line in cleaned.splitlines()
        ]

    seen: set[str] = set()
    unique = []
    for item in items:
        stripped = item.strip()
        if stripped and stripped.casefold() not in seen:
            seen.add(stripped.casefold())
            unique.append(stripped)
    return unique


def _from_json(text: str) -> list[str] | None:
    """Try to read the response as a JSON array of strings."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(parsed, list):
        return None
    return [
        item if isinstance(item, str) else str(item)
        for item in parsed
    ]
