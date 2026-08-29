"""PII detection, harmful content, and refusal detection."""

from __future__ import annotations

import pytest
from ragrank.dataset import DataNode
from ragrank.llm import FakeLLM
from ragrank.metric import (
    SAFETY_METRICS,
    Answered,
    CostTier,
    PIIFree,
    answered,
    find_pii,
    pii_free,
    safety,
)
from ragrank.metric._safety.pii import luhn


def node(
    response: str, context: list[str] | None = None
) -> DataNode:
    return DataNode(
        question="q", context=context or ["c"], response=response
    )


# ------------------------------ luhn ------------------------------


@pytest.mark.parametrize(
    "number",
    ["4111111111111111", "5500005555555559", "378282246310005"],
)
def test_luhn_accepts_real_card_numbers(number: str) -> None:
    assert luhn(number)


@pytest.mark.parametrize(
    "number",
    ["4111111111111112", "1234567890123456", "", "abc", "411111"],
)
def test_luhn_rejects_everything_else(number: str) -> None:
    assert not luhn(number)


# ------------------------------ pii ------------------------------


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("write to ada@example.com", "email"),
        ("ssn 123-45-6789", "ssn"),
        ("call 555-123-4567", "phone"),
        ("call (555) 123-4567", "phone"),
        ("host 192.168.1.1", "ip_address"),
        ("card 4111 1111 1111 1111", "credit_card"),
    ],
)
def test_find_pii_spots_each_kind(text: str, kind: str) -> None:
    assert kind in find_pii(text)


@pytest.mark.parametrize(
    "text",
    [
        "the answer is 42",
        "order number 1234 5678 9012 3456",
        "version 1.2.3.4.5",
        "released in 1843 by Ada",
        "call me at extension 4567",
    ],
)
def test_find_pii_leaves_ordinary_text_alone(text: str) -> None:
    assert find_pii(text) == {}


def test_a_long_number_that_fails_luhn_is_not_a_card() -> None:
    assert "credit_card" not in find_pii("id 1234567890123456")


def test_pii_free_scores_one_when_nothing_is_found() -> None:
    result = pii_free.score(node("Ada wrote it in 1843."))
    assert result.score == 1.0
    assert result.metadata["found"] == {}
    assert result.reason is None


def test_pii_free_scores_zero_and_says_what_it_found() -> None:
    result = pii_free.score(node("mail ada@example.com"))
    assert result.score == 0.0
    assert result.metadata["found"]["email"] == ["ada@example.com"]
    assert "email" in result.reason


def test_pii_free_gates_on_finding_anything() -> None:
    assert pii_free.threshold == 1.0
    assert pii_free.score(node("clean")).passed is True
    assert pii_free.score(node("a@b.com")).passed is False


def test_the_scan_can_be_restricted_to_some_kinds() -> None:
    only_ssn = PIIFree(kinds=["ssn"])
    assert only_ssn.score(node("mail a@b.com")).score == 1.0
    assert only_ssn.score(node("ssn 123-45-6789")).score == 0.0


def test_the_context_can_be_scanned_too() -> None:
    checked = PIIFree(check_context=True)
    leaky = node("nothing here", context=["ada@example.com"])

    assert pii_free.score(leaky).score == 1.0
    assert checked.score(leaky).score == 0.0
    assert checked.required_columns == {"response", "context"}


def test_duplicate_matches_are_reported_once() -> None:
    result = pii_free.score(node("a@b.com and a@b.com again"))
    assert result.metadata["found"]["email"] == ["a@b.com"]


def test_pii_free_costs_nothing() -> None:
    assert pii_free.cost_tier is CostTier.FREE
    assert pii_free.required_columns == {"response"}


# ---------------------------- answered ----------------------------


@pytest.mark.parametrize(
    "response",
    [
        "I cannot help with that.",
        "I can't answer that question.",
        "I'm sorry, but I cannot provide that information.",
        "As an AI language model, I must decline.",
        "I don't have enough information to answer.",
        "The provided context does not contain that information.",
        "I don't know.",
        "We are unable to assist with this.",
    ],
)
def test_refusals_are_detected(response: str) -> None:
    assert answered.score(node(response)).score == 0.0


@pytest.mark.parametrize(
    "response",
    [
        "Ada Lovelace wrote the first program.",
        "The context says it was 1843.",
        "I cannot overstate how important this was: it was the first.",
        "Paris. I don't know why you ask, but it is Paris.",
    ],
)
def test_real_answers_are_not_refusals(response: str) -> None:
    assert answered.score(node(response)).score == 1.0


def test_a_refusal_records_the_phrase_that_matched() -> None:
    result = answered.score(node("I cannot help with that."))
    assert "cannot help" in result.metadata["refusal"].lower()
    assert "refused" in result.reason


def test_an_answer_records_no_refusal() -> None:
    result = answered.score(node("It was 1843."))
    assert "refusal" not in result.metadata
    assert result.reason is None


def test_an_empty_response_abstains_rather_than_guessing() -> None:
    result = answered.score(node("   "))
    assert result.score is None
    assert result.error


def test_a_late_caveat_does_not_make_an_answer_a_refusal() -> None:
    long_answer = (
        "Ada Lovelace wrote the first published algorithm, in her "
        "notes on the Analytical Engine. "
        * 4
        + "I cannot confirm the exact date."
    )
    assert answered.score(node(long_answer)).score == 1.0


def test_the_search_window_is_configurable() -> None:
    late = "x" * 300 + " I cannot help with that."
    assert answered.score(node(late)).score == 1.0
    assert Answered(window=400).score(node(late)).score == 0.0


def test_answered_costs_nothing() -> None:
    assert answered.cost_tier is CostTier.FREE


# ----------------------------- safety -----------------------------


def test_safety_maps_the_rubric_letters() -> None:
    for letter, expected in [("A", 1.0), ("B", 0.5), ("C", 0.0)]:
        judged = safety.with_llm(FakeLLM(responses=[letter]))
        assert judged.score(node("anything")).score == expected


def test_safety_needs_only_the_response() -> None:
    assert safety.required_columns == {"response"}
    assert safety.cost_tier is CostTier.LLM


def test_an_unreadable_verdict_is_not_fatal() -> None:
    judged = safety.with_llm(FakeLLM(responses=["no idea"]))
    result = judged.score(node("anything"))
    assert result.score is None
    assert result.error


def test_the_preset_holds_all_three() -> None:
    assert SAFETY_METRICS == [pii_free, answered, safety]
