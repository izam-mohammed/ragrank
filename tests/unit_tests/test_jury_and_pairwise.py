"""Tests for committee judging and pairwise comparison."""

from __future__ import annotations

import pytest
from ragrank import evaluate
from ragrank.dataset import DataNode, Dataset
from ragrank.evaluation import RunConfig
from ragrank.llm import FakeLLM
from ragrank.metric import Jury, LLMJudge, Pairwise

SERIAL = RunConfig(show_progress=False, max_workers=1)
RUBRIC = {"A": 1.0, "B": 0.5, "C": 0.0}

NODE = DataNode(
    question="Capital of France?",
    context=["Paris is the capital of France."],
    response="Paris",
    reference="Paris is the capital",
)


def judge(name: str, verdict: str) -> LLMJudge:
    """A judge that always returns one verdict."""
    return LLMJudge(
        judge_name=name,
        instructions="x",
        rubric=dict(RUBRIC),
        llm=FakeLLM(responses=[verdict]),
    )


# --------------------------- Jury ---------------------------


def test_jury_combines_its_judges() -> None:
    """Median of the panel."""
    panel = Jury(
        judges=[judge("a", "A"), judge("b", "B"), judge("c", "A")]
    )
    assert panel.score(NODE).score == pytest.approx(1.0)


def test_jury_mean_aggregation() -> None:
    """Mean is available where the average matters."""
    panel = Jury(
        judges=[judge("a", "A"), judge("b", "C")],
        aggregation="mean",
    )
    assert panel.score(NODE).score == pytest.approx(0.5)


def test_every_judge_is_recorded_even_with_a_shared_name() -> None:
    """Verdicts are a list, so no judge is silently dropped."""
    panel = Jury(
        judges=[
            judge("same", "A"),
            judge("same", "B"),
            judge("other", "A"),
        ]
    )
    verdicts = panel.score(NODE).metadata["verdicts"]
    assert len(verdicts) == 3
    assert [item["score"] for item in verdicts] == [1.0, 0.5, 1.0]


def test_disagreement_is_reported() -> None:
    """Where judges argue is where the rubric is ambiguous."""
    split = Jury(judges=[judge("a", "A"), judge("b", "C")])
    assert split.score(NODE).metadata[
        "disagreement"
    ] == pytest.approx(1.0)

    agreed = Jury(judges=[judge("a", "A"), judge("b", "A")])
    assert agreed.score(NODE).metadata[
        "disagreement"
    ] == pytest.approx(0.0)


def test_jury_survives_one_broken_judge() -> None:
    """A judge that gives nothing usable does not sink the panel."""
    broken = LLMJudge(
        judge_name="broken",
        instructions="x",
        rubric=dict(RUBRIC),
        llm=FakeLLM(responses=["banana"]),
        max_retries=0,
    )
    panel = Jury(judges=[judge("a", "A"), broken])
    result = panel.score(NODE)
    assert result.score == pytest.approx(1.0)
    assert result.metadata["verdicts"][1]["score"] is None


def test_jury_abstains_when_no_judge_answers() -> None:
    """Nothing usable means no score."""
    broken = LLMJudge(
        judge_name="b",
        instructions="x",
        rubric=dict(RUBRIC),
        llm=FakeLLM(responses=["banana"]),
        max_retries=0,
    )
    result = Jury(judges=[broken]).score(NODE)
    assert result.score is None
    assert "no judge" in result.error


def test_jury_required_columns_is_the_union() -> None:
    """The runner must validate for every member."""
    panel = Jury(
        judges=[
            LLMJudge(
                judge_name="a",
                instructions="x",
                input_fields=["question"],
            ),
            LLMJudge(
                judge_name="b",
                instructions="x",
                input_fields=["response", "reference"],
            ),
        ]
    )
    assert panel.required_columns == {
        "question",
        "response",
        "reference",
    }


def test_jury_lends_the_runs_llm_to_bare_judges() -> None:
    """A panel assembled without models still works in a run."""
    panel = Jury(
        judges=[
            LLMJudge(
                judge_name="a",
                instructions="x",
                rubric=dict(RUBRIC),
            ),
            LLMJudge(
                judge_name="b",
                instructions="x",
                rubric=dict(RUBRIC),
            ),
        ],
        jury_name="Panel",
    )
    dataset = Dataset(
        question=["q"], context=[["c"]], response=["r"]
    )
    result = evaluate(
        dataset,
        llm=FakeLLM(responses=["A"]),
        metrics=[panel],
        run_config=SERIAL,
    )
    assert result.scores == [[1.0]]


# --------------------------- Pairwise ---------------------------


def prefers_paris(prompt: str) -> str:
    """Prefer whichever slot holds the short 'Paris' answer."""
    answer_a = (
        prompt.rsplit("answer_a:", 1)[1].split("\n")[0].strip()
    )
    return "A" if answer_a == "Paris" else "B"


def test_pairwise_win_is_consistent_both_ways() -> None:
    """A real win survives swapping the order."""
    metric = Pairwise(llm=FakeLLM(response_fn=prefers_paris))
    result = metric.score(NODE)
    assert result.score == 1.0
    assert result.metadata["position_bias"] is False


def test_pairwise_loss_is_consistent_both_ways() -> None:
    """So does a real loss."""
    metric = Pairwise(
        llm=FakeLLM(
            response_fn=lambda p: "B"
            if p.rsplit("answer_a:", 1)[1].split("\n")[0].strip()
            == "Paris"
            else "A"
        )
    )
    assert metric.score(NODE).score == 0.0


def test_position_bias_is_detected_not_rewarded() -> None:
    """A judge that always picks the first answer wins nothing.

    This is the reason the comparison is run twice.
    """
    biased = Pairwise(llm=FakeLLM(responses=["A"]))
    result = biased.score(NODE)
    assert result.score == 0.5
    assert result.metadata["position_bias"] is True


def test_pairwise_judges_each_pair_twice() -> None:
    """Two calls per row, by design."""
    llm = FakeLLM(response_fn=prefers_paris)
    Pairwise(llm=llm).score(NODE)
    assert len(llm.prompts) == 2


def test_pairwise_needs_a_baseline() -> None:
    """Nothing to compare against means no score."""
    result = Pairwise(llm=FakeLLM(responses=["A"])).score(
        DataNode(question="q", context=["c"], response="r")
    )
    assert result.score is None
    assert "reference" in result.error


def test_pairwise_declares_its_columns() -> None:
    """Including whichever baseline field was chosen."""
    assert Pairwise().required_columns == {
        "question",
        "response",
        "reference",
    }
    assert Pairwise(baseline_field="context").required_columns == {
        "question",
        "response",
        "context",
    }


def test_pairwise_abstains_on_an_unusable_verdict() -> None:
    """A judge that says nothing readable gives no comparison."""
    result = Pairwise(
        llm=FakeLLM(responses=["banana"]), max_retries=0
    ).score(NODE)
    assert result.score is None


def test_jury_calls_are_counted_in_usage() -> None:
    """A composite metric's calls must reach the token accounting.

    The runner used to leave non-LLMMetric metrics unbound, so a Jury
    never received the run's model at all and its calls were invisible.
    """
    panel = Jury(
        judges=[
            LLMJudge(
                judge_name="a",
                instructions="x",
                rubric=dict(RUBRIC),
            ),
            LLMJudge(
                judge_name="b",
                instructions="x",
                rubric=dict(RUBRIC),
            ),
        ]
    )
    dataset = Dataset(
        question=["q"], context=[["c"]], response=["r"]
    )
    result = evaluate(
        dataset,
        llm=FakeLLM(responses=["A"]),
        metrics=[panel],
        run_config=SERIAL,
    )
    assert result.usage.calls == 2
