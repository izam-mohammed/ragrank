"""Tests for per-chunk context scoring, and the issue #36 prompt fix."""

from __future__ import annotations

import pytest
from ragrank.dataset import DataNode
from ragrank.evaluation import RunConfig
from ragrank.llm import FakeLLM
from ragrank.metric import context_relevancy
from ragrank.prompt._prompts import (
    CONTEXT_RELEVANCY_PROMPT,
    CONTEXT_RELEVANCY_RUBRIC,
)

SERIAL = RunConfig(show_progress=False, max_workers=1)

THREE_CHUNKS = DataNode(
    question="What is the capital of France?",
    context=[
        "Paris is the capital of France.",
        "Cats sleep a lot.",
        "France is in Europe.",
    ],
    response="Paris.",
)


def verdicts(prompt: str) -> str:
    """Judge each chunk differently, keyed on its content."""
    if "Paris is the capital" in prompt:
        return "A"
    if "Cats" in prompt:
        return "C"
    return "B"


# ------------------------- issue #36 -------------------------


def test_issue_36_prompt_does_not_take_the_response() -> None:
    """Context relevancy is about (question, context) only.

    Feeding the response in leaks generation quality into what is
    supposed to be a retrieval metric.
    """
    assert CONTEXT_RELEVANCY_PROMPT.input_keys == [
        "question",
        "context",
    ]


def test_issue_36_examples_are_single_chunks() -> None:
    """Every example's context was silently concatenated before.

    Each context list held two adjacent string literals with no comma,
    so Python joined them and every 'two chunk' example demonstrated
    one chunk.
    """
    for example in CONTEXT_RELEVANCY_PROMPT.examples:
        assert isinstance(example["context"], str)
        assert example["relevancy"] in CONTEXT_RELEVANCY_RUBRIC


def test_issue_36_examples_cover_the_whole_rubric() -> None:
    """Few-shot examples should demonstrate each verdict."""
    shown = {
        example["relevancy"]
        for example in CONTEXT_RELEVANCY_PROMPT.examples
    }
    assert shown == set(CONTEXT_RELEVANCY_RUBRIC)


# ------------------------- chunkwise -------------------------


def test_each_chunk_is_judged_separately() -> None:
    """One prompt per chunk, and the verdicts are kept."""
    llm = FakeLLM(response_fn=verdicts)
    result = context_relevancy.model_copy(update={"llm": llm}).score(
        THREE_CHUNKS
    )

    assert len(llm.prompts) == 3
    assert result.metadata["chunk_scores"] == [1.0, 0.0, 0.5]
    assert result.score == pytest.approx(0.5)


def test_one_bad_chunk_is_visible_not_smoothed_away() -> None:
    """The point of per-chunk scoring: you can see which chunk failed."""
    llm = FakeLLM(response_fn=verdicts)
    result = context_relevancy.model_copy(update={"llm": llm}).score(
        THREE_CHUNKS
    )
    assert result.metadata["chunk_scores"][1] == 0.0


def test_an_unreadable_chunk_does_not_sink_the_row() -> None:
    """Other chunks still count when one answer is unusable."""
    llm = FakeLLM(
        response_fn=lambda p: "banana" if "Cats" in p else "A"
    )
    result = context_relevancy.model_copy(
        update={"llm": llm, "max_retries": 0}
    ).score(THREE_CHUNKS)

    assert result.metadata["chunk_scores"] == [1.0, None, 1.0]
    assert result.score == pytest.approx(1.0)


def test_all_chunks_unreadable_abstains() -> None:
    """If nothing could be judged, the row has no score."""
    result = context_relevancy.model_copy(
        update={
            "llm": FakeLLM(responses=["banana"]),
            "max_retries": 0,
        }
    ).score(THREE_CHUNKS)

    assert result.score is None
    assert result.error is not None
    assert result.metadata["chunk_scores"] == [None, None, None]


def test_empty_context_abstains() -> None:
    """Nothing retrieved means nothing to judge."""
    llm = FakeLLM()
    result = context_relevancy.model_copy(update={"llm": llm}).score(
        DataNode(question="q", context=[], response="r")
    )
    assert result.score is None
    assert result.error == "no context to judge"
    assert llm.prompts == []


def test_the_prompt_shows_one_chunk_at_a_time() -> None:
    """Each prompt must contain its own chunk and not the others."""
    llm = FakeLLM(response_fn=verdicts)
    context_relevancy.model_copy(update={"llm": llm}).score(
        THREE_CHUNKS
    )
    combined = "\n".join(llm.prompts)
    assert combined.count("Cats sleep a lot.") == 1
