"""Tests for the deterministic ranking metrics.

Expected values are worked out by hand in the docstrings so a future
change that breaks the maths is obvious rather than plausible.
"""

from __future__ import annotations

from math import log2

import pytest
from ragrank.dataset import DataNode
from ragrank.metric import (
    MAP,
    MRR,
    NDCG,
    HitRate,
    PrecisionAtK,
    RecallAtK,
    hit_rate,
    mean_average_precision,
    mrr,
    ndcg,
    precision_at_k,
    recall_at_k,
)

ALL_RANKING_METRICS = [
    hit_rate,
    mrr,
    precision_at_k,
    recall_at_k,
    mean_average_precision,
    ndcg,
]


def node(
    retrieved: list[str] | None = None,
    reference: list[str] | None = None,
) -> DataNode:
    """Build a data node carrying document ids."""
    return DataNode(
        question="q",
        context=["c"],
        response="r",
        retrieved_ids=retrieved,
        reference_ids=reference,
    )


#: retrieved d3 d1 d7 d2, relevant {d1, d2} -- hits at ranks 2 and 4.
MIXED = node(["d3", "d1", "d7", "d2"], ["d1", "d2"])


def test_hit_rate() -> None:
    """1.0 when any relevant document appears."""
    assert hit_rate.score(MIXED).score == 1.0
    assert hit_rate.score(node(["x", "y"], ["d1"])).score == 0.0


def test_mrr_uses_the_first_hit() -> None:
    """First hit is at rank 2, so MRR = 1/2."""
    assert mrr.score(MIXED).score == pytest.approx(0.5)
    assert mrr.score(node(["d1", "x"], ["d1"])).score == 1.0
    assert mrr.score(node(["x"], ["d1"])).score == 0.0


def test_precision() -> None:
    """2 relevant of 4 retrieved = 0.5."""
    assert precision_at_k.score(MIXED).score == pytest.approx(0.5)


def test_recall() -> None:
    """Both relevant documents were found = 1.0."""
    assert recall_at_k.score(MIXED).score == pytest.approx(1.0)
    assert recall_at_k.score(
        node(["d1"], ["d1", "d2"])
    ).score == pytest.approx(0.5)


def test_average_precision() -> None:
    """Hits at ranks 2 and 4: (1/2 + 2/4) / 2 = 0.5."""
    assert mean_average_precision.score(
        MIXED
    ).score == pytest.approx(0.5)


def test_ndcg_against_hand_computed_value() -> None:
    """DCG = 1/log2(3) + 1/log2(5); ideal = 1/log2(2) + 1/log2(3)."""
    gain = 1 / log2(3) + 1 / log2(5)
    ideal = 1 / log2(2) + 1 / log2(3)
    assert ndcg.score(MIXED).score == pytest.approx(gain / ideal)


def test_perfect_ranking_scores_one_everywhere() -> None:
    """A perfect retrieval maxes out every metric."""
    perfect = node(["d1", "d2"], ["d1", "d2"])
    for metric in ALL_RANKING_METRICS:
        assert metric.score(perfect).score == pytest.approx(
            1.0
        ), metric.name


def test_total_miss_scores_zero_everywhere() -> None:
    """Retrieving nothing relevant scores zero, not None."""
    missed = node(["x", "y"], ["d1"])
    for metric in ALL_RANKING_METRICS:
        assert metric.score(missed).score == pytest.approx(
            0.0
        ), metric.name


@pytest.mark.parametrize(
    ("k", "expected_recall", "expected_hit"),
    [(1, 0.0, 0.0), (2, 0.5, 1.0), (4, 1.0, 1.0)],
)
def test_k_truncates_the_retrieved_list(
    k: int, expected_recall: float, expected_hit: float
) -> None:
    """@k considers only the top k results."""
    assert RecallAtK(k=k).score(MIXED).score == pytest.approx(
        expected_recall
    )
    assert HitRate(k=k).score(MIXED).score == pytest.approx(
        expected_hit
    )


def test_k_appears_in_the_metric_name() -> None:
    """Names must disambiguate Recall@1 from Recall@5 in a report."""
    assert RecallAtK(k=5).name == "Recall@5"
    assert HitRate(k=3).name == "Hit Rate@3"
    assert MRR().name == "MRR"
    assert NDCG(k=10).name == "NDCG@10"
    assert PrecisionAtK(k=2).name == "Precision@2"
    assert MAP(k=2).name == "MAP@2"


@pytest.mark.parametrize("metric", ALL_RANKING_METRICS)
def test_abstains_without_ids(metric: object) -> None:
    """Missing ids means None, not a fabricated score."""
    result = metric.score(node())
    assert result.score is None
    assert result.error is not None


@pytest.mark.parametrize("metric", ALL_RANKING_METRICS)
def test_abstains_when_nothing_is_relevant(metric: object) -> None:
    """An empty reference set makes every ranking metric undefined."""
    assert metric.score(node(["d1"], [])).score is None


@pytest.mark.parametrize("metric", ALL_RANKING_METRICS)
def test_declares_required_columns(metric: object) -> None:
    """The runner validates on these before spending anything."""
    assert metric.required_columns == {
        "retrieved_ids",
        "reference_ids",
    }


@pytest.mark.parametrize("metric", ALL_RANKING_METRICS)
def test_needs_no_llm(metric: object) -> None:
    """Ranking metrics must never touch a model."""
    assert metric.llm is None
    assert metric.prompt is None


def test_precision_abstains_on_empty_retrieval() -> None:
    """Precision over zero retrieved documents is undefined."""
    assert precision_at_k.score(node([], ["d1"])).score is None


def test_recall_is_zero_on_empty_retrieval() -> None:
    """Recall over zero retrieved documents is legitimately zero."""
    assert recall_at_k.score(node([], ["d1"])).score == 0.0


def test_duplicate_retrieved_ids_do_not_inflate_recall() -> None:
    """Returning the same hit twice is not two hits."""
    assert recall_at_k.score(
        node(["d1", "d1"], ["d1", "d2"])
    ).score == pytest.approx(0.5)
