"""Attributing the correct part of an answer to retrieval."""

from __future__ import annotations

from ragrank.dataset import DataNode
from ragrank.llm import FakeLLM
from ragrank.metric import CostTier, context_reliance

CLAIMS = "- Ada wrote it.\n- It was 1843."


def node(
    response: str = "Ada wrote it in 1843.",
    context: list[str] | None = None,
    reference: str | None = "Ada Lovelace wrote it in 1843.",
) -> DataNode:
    return DataNode(
        question="who wrote it",
        context=["Ada Lovelace wrote it."]
        if context is None
        else context,
        response=response,
        reference=reference,
    )


def scripted(*, correct: list[str], supported: list[str]) -> FakeLLM:
    """A judge that answers extraction, then correctness, then support.

    Verdicts are popped per call, so each claim can be given a
    different pair of answers.
    """
    correct_left = list(correct)
    supported_left = list(supported)

    def answer(prompt: str) -> str:
        if prompt.startswith("Claim Extraction"):
            return CLAIMS
        # The reference pass names the reference; the context pass
        # names the retrieved chunk.
        if "Ada Lovelace wrote it in 1843." in prompt:
            return correct_left.pop(0)
        return supported_left.pop(0)

    return FakeLLM(response_fn=answer)


def judged(llm: FakeLLM):
    return context_reliance.with_llm(llm)


def test_everything_correct_came_from_the_context() -> None:
    llm = scripted(correct=["A", "A"], supported=["A", "A"])
    result = judged(llm).score(node())

    assert result.score == 1.0
    assert result.metadata["self_knowledge"] == 0.0
    assert result.metadata["correct_count"] == 2


def test_correct_but_unsupported_is_the_models_own_knowledge() -> (
    None
):
    llm = scripted(correct=["A", "A"], supported=["B", "B"])
    result = judged(llm).score(node())

    assert result.score == 0.0
    assert result.metadata["self_knowledge"] == 1.0


def test_a_mixed_answer_lands_in_between() -> None:
    llm = scripted(correct=["A", "A"], supported=["A", "B"])
    result = judged(llm).score(node())

    assert result.score == 0.5
    assert result.metadata["self_knowledge"] == 0.5


def test_wrong_claims_are_not_held_against_the_retriever() -> None:
    """An incorrect claim is faithfulness's problem, not retrieval's."""
    llm = scripted(correct=["A", "B"], supported=["A"])
    result = judged(llm).score(node())

    assert result.score == 1.0
    assert result.metadata["claim_count"] == 2
    assert result.metadata["correct_count"] == 1


def test_a_wrong_claim_is_never_checked_for_support() -> None:
    """The second pass is skipped, which is the point of the ordering."""
    llm = scripted(correct=["B", "B"], supported=[])
    result = judged(llm).score(node())

    assert result.score is None
    assert "no correct claim" in result.error
    verdicts = result.metadata["claims"]
    assert all(item["supported"] is None for item in verdicts)


def test_per_claim_verdicts_are_recorded() -> None:
    llm = scripted(correct=["A", "A"], supported=["A", "B"])
    verdicts = judged(llm).score(node()).metadata["claims"]

    assert [item["claim"] for item in verdicts] == [
        "Ada wrote it.",
        "It was 1843.",
    ]
    assert verdicts[0]["supported"] == 1.0
    assert verdicts[1]["supported"] == 0.0


def test_no_reference_means_no_judgement() -> None:
    llm = FakeLLM(responses=["A"])
    result = judged(llm).score(node(reference=None))

    assert result.score is None
    assert "no reference" in result.error


def test_a_blank_reference_is_the_same_as_none() -> None:
    llm = FakeLLM(responses=["A"])
    result = judged(llm).score(node(reference="   "))

    assert result.score is None
    assert "no reference" in result.error


def test_no_context_means_no_attribution() -> None:
    llm = FakeLLM(responses=["A"])
    result = judged(llm).score(node(context=[]))

    assert result.score is None
    assert "no context" in result.error


def test_a_response_with_no_claims_abstains() -> None:
    llm = FakeLLM(response_fn=lambda prompt: "")
    result = judged(llm).score(node())

    assert result.score is None
    assert "no verifiable claims" in result.error


def test_it_declares_what_it_needs_and_what_it_costs() -> None:
    assert context_reliance.required_columns == {
        "context",
        "response",
        "reference",
    }
    assert context_reliance.cost_tier is CostTier.LLM_HEAVY


def test_the_claim_cap_still_applies() -> None:
    capped = context_reliance.model_copy(update={"max_claims": 1})
    llm = scripted(correct=["A"], supported=["A"])
    result = capped.with_llm(llm).score(node())

    assert result.metadata["claim_count"] == 1
