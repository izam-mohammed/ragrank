"""Tests for claim-level faithfulness and reference correctness."""

from __future__ import annotations

import pytest
from ragrank.dataset import DataNode
from ragrank.llm import FakeLLM
from ragrank.metric import (
    RAG_TRIAD,
    ClaimMetric,
    Correctness,
    Faithfulness,
    correctness,
    faithfulness,
)
from ragrank.metric.parse import parse_list

GROUNDED = "The Eiffel Tower is in Paris."
INVENTED = "The Eiffel Tower was built in 1750."

NODE = DataNode(
    question="Where is the Eiffel Tower?",
    context=[
        "The Eiffel Tower stands on the Champ de Mars in Paris."
    ],
    response=f"{GROUNDED} {INVENTED}",
    reference="Paris",
)


def judge(prompt: str) -> str:
    """Extract two claims, ground only the first."""
    if prompt.startswith("Claim Extraction"):
        return f'["{GROUNDED}", "{INVENTED}"]'
    claim = prompt.rsplit("claim:", 1)[1].split("\n")[0].strip()
    return "A" if "Paris" in claim else "C"


def with_llm(metric: object, llm: FakeLLM) -> object:
    """Bind a fake LLM to a metric."""
    return metric.model_copy(update={"llm": llm})


# --------------------------- parse_list ---------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('["a", "b"]', ["a", "b"]),
        ('Here you go:\n["a"]', ["a"]),
        ("1. first\n2. second", ["first", "second"]),
        ("- one\n* two\n• three", ["one", "two", "three"]),
        ("bare line\nanother", ["bare line", "another"]),
        ('```json\n["x"]\n```', ["x"]),
        ("[]", []),
        ("", []),
        ("   ", []),
        ("[1, 2]", ["1", "2"]),
        ("[not valid json", ["[not valid json"]),
    ],
)
def test_parse_list(text: str, expected: list[str]) -> None:
    """Models emit lists four different ways; accept all of them."""
    assert parse_list(text) == expected


def test_parse_list_deduplicates_case_insensitively() -> None:
    """The same claim twice is one claim."""
    assert parse_list("Paris\nparis\nPARIS") == ["Paris"]


def test_parse_list_never_raises() -> None:
    """Whatever came back, it is not an exception."""
    assert parse_list(None) == []


# --------------------------- faithfulness ---------------------------


def test_faithfulness_is_the_supported_ratio() -> None:
    """One grounded claim of two is 0.5."""
    result = with_llm(
        faithfulness, FakeLLM(response_fn=judge)
    ).score(NODE)
    assert result.score == pytest.approx(0.5)


def test_faithfulness_names_the_offending_claim() -> None:
    """A bad score must point at the sentence that caused it."""
    result = with_llm(
        faithfulness, FakeLLM(response_fn=judge)
    ).score(NODE)
    claims = {
        item["claim"]: item["supported"]
        for item in result.metadata["claims"]
    }
    assert claims[GROUNDED] == 1.0
    assert claims[INVENTED] == 0.0
    assert result.metadata["claim_count"] == 2


def test_fully_grounded_response_scores_one() -> None:
    """Everything supported is a perfect score."""
    llm = FakeLLM(
        response_fn=lambda p: f'["{GROUNDED}"]'
        if p.startswith("Claim Extraction")
        else "A"
    )
    assert with_llm(faithfulness, llm).score(NODE).score == 1.0


def test_fully_invented_response_scores_zero() -> None:
    """Nothing supported is zero, not an abstention."""
    llm = FakeLLM(
        response_fn=lambda p: f'["{INVENTED}"]'
        if p.startswith("Claim Extraction")
        else "C"
    )
    assert with_llm(faithfulness, llm).score(NODE).score == 0.0


def test_a_contradicted_claim_counts_against() -> None:
    """Verdict C is unsupported, same as B."""
    llm = FakeLLM(
        response_fn=lambda p: f'["{GROUNDED}", "{INVENTED}"]'
        if p.startswith("Claim Extraction")
        else "B"
    )
    assert with_llm(faithfulness, llm).score(NODE).score == 0.0


def test_no_claims_abstains_rather_than_scoring_zero() -> None:
    """ "Thanks for asking!" is not a hallucination."""
    result = with_llm(faithfulness, FakeLLM(responses=["[]"])).score(
        DataNode(
            question="q",
            context=["c"],
            response="Thanks for asking!",
        )
    )
    assert result.score is None
    assert "no verifiable claims" in result.error
    assert result.metadata["claims"] == []


def test_empty_context_abstains_without_prompting() -> None:
    """Nothing to verify against, so do not spend anything."""
    llm = FakeLLM()
    result = with_llm(faithfulness, llm).score(
        DataNode(question="q", context=[], response="something")
    )
    assert result.score is None
    assert "no source text" in result.error
    assert llm.prompts == []


def test_unusable_verdicts_abstain() -> None:
    """If no claim could be judged, there is no ratio."""
    llm = FakeLLM(
        response_fn=lambda p: f'["{GROUNDED}"]'
        if p.startswith("Claim Extraction")
        else "banana"
    )
    result = with_llm(
        faithfulness.model_copy(update={"max_retries": 0}), llm
    ).score(NODE)
    assert result.score is None
    assert "no claim could be verified" in result.error


def test_partially_unusable_verdicts_still_score() -> None:
    """One unreadable verdict does not cost the other claims."""

    def flaky(prompt: str) -> str:
        if prompt.startswith("Claim Extraction"):
            return f'["{GROUNDED}", "{INVENTED}"]'
        claim = prompt.rsplit("claim:", 1)[1].split("\n")[0].strip()
        return "A" if "Paris" in claim else "banana"

    result = with_llm(
        faithfulness.model_copy(update={"max_retries": 0}),
        FakeLLM(response_fn=flaky),
    ).score(NODE)
    assert result.score == 1.0
    assert result.metadata["verified_count"] == 1
    assert result.metadata["claim_count"] == 2


def test_max_claims_caps_the_spend() -> None:
    """A very long response must not run away with the budget."""
    many = "[" + ",".join(f'"claim {i}"' for i in range(100)) + "]"
    llm = FakeLLM(
        response_fn=lambda p: many
        if p.startswith("Claim Extraction")
        else "A"
    )
    result = with_llm(
        faithfulness.model_copy(update={"max_claims": 5}), llm
    ).score(NODE)
    assert result.metadata["claim_count"] == 5
    assert len(llm.prompts) == 6  # 1 extraction + 5 verifications


def test_faithfulness_declares_its_columns() -> None:
    """It needs no reference — it is a grounding check, not correctness."""
    assert faithfulness.required_columns == {"context", "response"}


def test_faithfulness_is_in_the_rag_triad() -> None:
    """The triad should include the hallucination leg."""
    assert faithfulness in RAG_TRIAD


def test_claim_metric_base_is_abstract_in_behaviour() -> None:
    """Subclasses must say what to decompose and what to check against."""

    class Incomplete(ClaimMetric):
        @property
        def name(self) -> str:
            return "Incomplete"

    with pytest.raises(NotImplementedError):
        Incomplete(llm=FakeLLM()).score(NODE)


# --------------------------- correctness ---------------------------


@pytest.mark.parametrize(
    ("verdict", "expected"), [("A", 1.0), ("B", 0.5), ("C", 0.0)]
)
def test_correctness_maps_verdicts(
    verdict: str, expected: float
) -> None:
    """Right, partly right, wrong."""
    result = with_llm(
        correctness, FakeLLM(responses=[verdict])
    ).score(NODE)
    assert result.score == expected


def test_correctness_needs_a_reference() -> None:
    """It is a reference-based metric and says so."""
    assert correctness.required_columns == {
        "question",
        "reference",
        "response",
    }


def test_correctness_shows_the_judge_the_reference() -> None:
    """The reference must actually reach the prompt."""
    llm = FakeLLM(responses=["A"])
    with_llm(correctness, llm).score(NODE)
    assert "Paris" in llm.prompts[0]


def test_correctness_rejects_an_off_rubric_answer() -> None:
    """A float is not a valid verdict here."""
    result = with_llm(
        Correctness(max_retries=0), FakeLLM(responses=["0.7"])
    ).score(NODE)
    assert result.score is None


def test_correctness_beats_string_matching_on_phrasing() -> None:
    """The reason this metric exists, stated as a test."""
    from ragrank.metric import exact_match

    node = DataNode(
        question="What is the capital of France?",
        context=["c"],
        response="The capital of France is Paris.",
        reference="Paris",
    )
    assert exact_match.score(node).score == 0.0
    assert (
        with_llm(correctness, FakeLLM(responses=["A"]))
        .score(node)
        .score
        == 1.0
    )


def test_faithfulness_and_correctness_are_different_questions() -> (
    None
):
    """A response can be grounded and wrong, or right and ungrounded."""
    assert Faithfulness().required_columns != (
        Correctness().required_columns
    )
