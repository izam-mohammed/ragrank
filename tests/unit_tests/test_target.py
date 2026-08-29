"""Running the system under test."""

from __future__ import annotations

import pytest

from ragrank import evaluate
from ragrank.dataset import Dataset, from_dict
from ragrank.llm import FakeLLM
from ragrank.metric import response_relevancy
from ragrank.target import (
    TargetError,
    TargetOutput,
    normalise_output,
    run_target,
)

QUESTIONS = ["who wrote it", "when was it"]


def simple(question: str) -> str:
    return f"answer to {question}"


def with_context(question: str) -> tuple[str, list[str]]:
    return f"answer to {question}", [f"chunk for {question}"]


def test_a_bare_string_is_an_answer_with_no_retrieval() -> None:
    output = normalise_output("hello")
    assert output.response == "hello"
    assert output.context == []


def test_a_tuple_is_read_as_response_and_context() -> None:
    output = normalise_output(("hi", ["a", "b"]))
    assert output.response == "hi"
    assert output.context == ["a", "b"]


def test_a_string_context_becomes_one_chunk() -> None:
    assert normalise_output(("hi", "one")).context == ["one"]


def test_a_mapping_is_read_by_key() -> None:
    output = normalise_output({
        "response": "hi",
        "context": ["a"],
        "retrieved_ids": ["d1"],
    })
    assert output.retrieved_ids == ["d1"]


def test_a_target_output_passes_straight_through() -> None:
    given = TargetOutput(response="hi")
    assert normalise_output(given) is given


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (("a", "b", "c"), "3"),
        ({"answer": "a"}, "'response' key"),
        (42, "not int"),
    ],
)
def test_unreadable_target_returns_are_named(value, match) -> None:
    with pytest.raises(TargetError, match=match):
        normalise_output(value)


def test_run_target_builds_a_dataset() -> None:
    dataset = run_target(QUESTIONS, with_context, max_workers=1)

    assert isinstance(dataset, Dataset)
    assert dataset.question == QUESTIONS
    assert dataset.response == [
        "answer to who wrote it",
        "answer to when was it",
    ]
    assert dataset.context == [
        ["chunk for who wrote it"],
        ["chunk for when was it"],
    ]


def test_run_target_keeps_question_order_when_concurrent() -> None:
    dataset = run_target(
        [str(index) for index in range(20)],
        simple,
        max_workers=8,
    )
    assert dataset.question == [str(index) for index in range(20)]
    assert dataset.response == [
        f"answer to {index}" for index in range(20)
    ]


def test_run_target_carries_references() -> None:
    dataset = run_target(
        QUESTIONS, simple, references=["a", "b"], max_workers=1
    )
    assert dataset.reference == ["a", "b"]


def test_run_target_rejects_mismatched_references() -> None:
    with pytest.raises(ValueError, match="must line up"):
        run_target(QUESTIONS, simple, references=["only one"])


def test_run_target_rejects_no_questions() -> None:
    with pytest.raises(ValueError, match="no questions"):
        run_target([], simple)


def test_run_target_reuses_a_datasets_questions() -> None:
    original = from_dict(
        {
            "question": QUESTIONS,
            "context": [["old"], ["old"]],
            "response": ["stale", "stale"],
            "reference": ["a", "b"],
        },
        return_as_dataset=True,
    )
    regenerated = run_target(original, with_context, max_workers=1)

    assert regenerated.question == QUESTIONS
    assert regenerated.reference == ["a", "b"]
    assert regenerated.response != original.response
    assert regenerated.context == [
        ["chunk for who wrote it"],
        ["chunk for when was it"],
    ]


def test_retrieved_ids_come_through_when_every_row_has_them() -> None:
    def with_ids(question: str) -> dict:
        return {
            "response": "a",
            "context": ["c"],
            "retrieved_ids": ["d1"],
        }

    dataset = run_target(QUESTIONS, with_ids, max_workers=1)
    assert dataset.retrieved_ids == [["d1"], ["d1"]]


def test_partial_retrieved_ids_are_dropped_rather_than_faked() -> None:
    def sometimes(question: str) -> dict:
        ids = ["d1"] if question == QUESTIONS[0] else None
        return {"response": "a", "context": ["c"], "retrieved_ids": ids}

    dataset = run_target(QUESTIONS, sometimes, max_workers=1)
    assert dataset.retrieved_ids is None


def test_a_failing_target_raises_by_default() -> None:
    def broken(question: str) -> str:
        raise RuntimeError("upstream is down")

    with pytest.raises(TargetError, match="upstream is down"):
        run_target(QUESTIONS, broken, max_workers=1)


def test_failures_can_be_skipped() -> None:
    def flaky(question: str) -> str:
        if question == QUESTIONS[0]:
            raise RuntimeError("nope")
        return "fine"

    dataset = run_target(
        QUESTIONS, flaky, max_workers=1, skip_failures=True
    )
    assert dataset.question == [QUESTIONS[1]]
    assert dataset.response == ["fine"]


def test_skipping_everything_is_still_an_error() -> None:
    def broken(question: str) -> str:
        raise RuntimeError("nope")

    with pytest.raises(TargetError, match="every question"):
        run_target(
            QUESTIONS, broken, max_workers=1, skip_failures=True
        )


def test_a_target_is_retried_before_giving_up() -> None:
    attempts = []

    def flaky(question: str) -> str:
        attempts.append(question)
        if len(attempts) < 3:
            raise RuntimeError("try again")
        return "eventually"

    dataset = run_target(
        ["only"], flaky, max_workers=1, max_retries=3, backoff=0.0
    )
    assert dataset.response == ["eventually"]
    assert len(attempts) == 3


def test_evaluate_runs_the_target_for_you() -> None:
    result = evaluate(
        QUESTIONS,
        target=with_context,
        llm=FakeLLM(responses=["0.8"]),
        metrics=[response_relevancy],
        run_config=None,
    )
    assert result.dataset.question == QUESTIONS
    assert result.scores == [[0.8, 0.8]]


def test_evaluate_rejects_questions_with_no_target() -> None:
    with pytest.raises(ValueError, match="no responses to score"):
        evaluate(QUESTIONS, metrics=[response_relevancy])
