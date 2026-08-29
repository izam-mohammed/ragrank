"""Edge cases and the not-implemented parts of the public surface."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from ragrank import evaluate
from ragrank.dataset import DataNode, Dataset
from ragrank.evaluation import RunConfig
from ragrank.llm import BaseLLM, LLMResult
from ragrank.metric import (
    RankingMetric,
    exact_match,
    levenshtein_ratio,
    response_relevancy,
    rouge_l,
    token_f1,
)
from ragrank.prompt import Prompt
from ragrank.utils.common import eval_cell


def node(response: str, reference: str | None = None) -> DataNode:
    """Build a one-off data node."""
    return DataNode(
        question="q",
        context=["c"],
        response=response,
        reference=reference,
    )


# --------------------- empty-text edge cases ---------------------


def test_levenshtein_ratio_of_two_empty_strings() -> None:
    """Two empty strings are identical, not undefined."""
    assert levenshtein_ratio.score(node("", "")).score == 1.0


def test_levenshtein_ratio_of_punctuation_only() -> None:
    """Normalisation can empty both sides."""
    assert levenshtein_ratio.score(node("...", "!!!")).score == 1.0


@pytest.mark.parametrize("metric", [token_f1, rouge_l])
def test_overlap_metrics_with_both_sides_empty(
    metric: object,
) -> None:
    """Empty vs empty is a perfect match."""
    assert metric.score(node("", "")).score == 1.0


@pytest.mark.parametrize("metric", [token_f1, rouge_l])
def test_overlap_metrics_with_one_side_empty(
    metric: object,
) -> None:
    """Empty vs non-empty is a total miss."""
    assert metric.score(node("", "something")).score == 0.0
    assert metric.score(node("something", "")).score == 0.0


def test_rouge_l_with_no_common_subsequence() -> None:
    """Disjoint token sets score zero."""
    assert (
        rouge_l.score(node("alpha beta", "gamma delta")).score == 0.0
    )


def test_exact_match_of_empty_strings() -> None:
    """Both empty is still a match."""
    assert exact_match.score(node("", "")).score == 1.0


# --------------------------- eval_cell ---------------------------


def test_eval_cell_rejects_a_parsed_non_list() -> None:
    """A tuple literal is not a context list."""
    assert eval_cell("[1, 2][0]") == "[1, 2][0]"


def test_eval_cell_stringifies_non_string_items() -> None:
    """Numbers in a context list become strings."""
    assert eval_cell("[1, 2.5, None]") == ["1", "2.5", "None"]


# --------------------- retries and backoff ---------------------


def test_transport_retry_sleeps_between_attempts() -> None:
    """Backoff must actually wait, and double each time."""
    attempts = {"count": 0}

    class Flaky(BaseLLM):
        @property
        def name(self) -> str:
            return "Flaky"

        def generate_text(self, text: str) -> LLMResult:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ConnectionError("temporary")
            return LLMResult(response="0.5")

    dataset = Dataset(
        question=["q"], context=[["c"]], response=["r"]
    )
    with patch("ragrank.evaluation.runner.sleep") as slept:
        result = evaluate(
            dataset,
            llm=Flaky(),
            metrics=[response_relevancy],
            run_config=RunConfig(
                show_progress=False,
                max_workers=1,
                max_retries=3,
                backoff=0.5,
            ),
        )

    assert result.scores == [[0.5]]
    assert [call.args[0] for call in slept.call_args_list] == [
        0.5,
        1.0,
    ]


def test_zero_backoff_does_not_sleep() -> None:
    """backoff=0 means retry immediately."""

    class Broken(BaseLLM):
        @property
        def name(self) -> str:
            return "Broken"

        def generate_text(self, text: str) -> LLMResult:
            raise ConnectionError("down")

    dataset = Dataset(
        question=["q"], context=[["c"]], response=["r"]
    )
    with patch("ragrank.evaluation.runner.sleep") as slept:
        evaluate(
            dataset,
            llm=Broken(),
            metrics=[response_relevancy],
            run_config=RunConfig(
                show_progress=False,
                max_workers=1,
                max_retries=2,
                backoff=0.0,
            ),
        )
    slept.assert_not_called()


# --------------------- not implemented surface ---------------------


def test_metric_save_and_load_are_not_implemented() -> None:
    """Declared but unbuilt; must fail clearly rather than silently."""
    with pytest.raises(NotImplementedError):
        response_relevancy.save()
    with pytest.raises(NotImplementedError):
        response_relevancy.load()


def test_prompt_save_and_load_are_not_implemented() -> None:
    """Same for prompts."""
    prompt = Prompt(
        name="P",
        instructions="i",
        input_keys=["question"],
        output_key="out",
    )
    with pytest.raises(NotImplementedError):
        prompt.save()
    with pytest.raises(NotImplementedError):
        prompt.load()


def test_ranking_metric_base_requires_a_rank_implementation() -> (
    None
):
    """The base class is abstract in behaviour, if not in form."""

    class Incomplete(RankingMetric):
        @property
        def name(self) -> str:
            return "Incomplete"

    with pytest.raises(NotImplementedError):
        Incomplete().score(
            DataNode(
                question="q",
                context=["c"],
                response="r",
                retrieved_ids=["d1"],
                reference_ids=["d1"],
            )
        )


def test_prompt_render_names_missing_keys() -> None:
    """The error must say what was needed and what was there."""
    prompt = Prompt(
        name="P",
        instructions="i",
        input_keys=["question", "reference"],
        output_key="out",
    )
    with pytest.raises(KeyError) as caught:
        prompt.render({"question": "q"})
    message = str(caught.value)
    assert "reference" in message
    assert "question" in message


def test_llm_metric_reason_hook_defaults_to_none() -> None:
    """Subclasses may override; the default adds no cost."""
    assert response_relevancy.reason(node("r"), 0.5, "raw") is None


def test_core_imports_without_the_provider_extras() -> None:
    """Only provider SDKs are optional; the core must not need them."""
    blocked = {
        "datasets": None,
        "openai": None,
        "langchain_core": None,
    }
    with patch.dict(sys.modules, blocked):
        from ragrank.metric import hit_rate, token_f1

        assert token_f1.name
        assert hit_rate.name
