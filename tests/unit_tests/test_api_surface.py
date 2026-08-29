"""Tests for the public entry points and remaining code paths."""

from __future__ import annotations

from pathlib import Path

import pytest
from ragrank import evaluate
from ragrank.dataset import (
    ColumnMap,
    DataNode,
    Dataset,
    from_csv,
    from_dataframe,
    from_dict,
)
from ragrank.evaluation import RunConfig, run_metrics
from ragrank.llm import FakeLLM
from ragrank.metric import (
    MAP,
    MRR,
    NDCG,
    ExactMatch,
    HitRate,
    InstructConfig,
    JsonValid,
    LevenshteinRatio,
    MetricType,
    PrecisionAtK,
    RecallAtK,
    RougeL,
    StringPresence,
    TokenF1,
    exact_match,
    response_relevancy,
)

SERIAL = RunConfig(show_progress=False, max_workers=1)


# --------------------- evaluate() input coercion ---------------------


def test_evaluate_accepts_a_plain_dict() -> None:
    """The quickstart shape must work."""
    result = evaluate(
        {
            "question": "q",
            "context": ["c"],
            "response": "r",
        },
        llm=FakeLLM(responses=["0.5"]),
        run_config=SERIAL,
    )
    assert result.scores == [[0.5]]


def test_evaluate_accepts_a_single_datanode() -> None:
    """A lone node is promoted to a one-row dataset."""
    result = evaluate(
        DataNode(question="q", context=["c"], response="r"),
        llm=FakeLLM(responses=["0.5"]),
        run_config=SERIAL,
    )
    assert len(result.dataset) == 1


def test_evaluate_accepts_a_bare_metric() -> None:
    """metrics= need not be a list."""
    result = evaluate(
        DataNode(question="q", context=["c"], response="r"),
        llm=FakeLLM(responses=["0.5"]),
        metrics=response_relevancy,
        run_config=SERIAL,
    )
    assert len(result.metrics) == 1


def test_evaluate_defaults_to_response_relevancy() -> None:
    """Omitting metrics picks a sensible default."""
    result = evaluate(
        DataNode(question="q", context=["c"], response="r"),
        llm=FakeLLM(responses=["0.5"]),
        run_config=SERIAL,
    )
    assert result.metrics[0].name == "Response Relevancy"


def test_run_metrics_is_usable_directly() -> None:
    """The runner is public and works without evaluate().

    It returns (results, usage) -- the usage half was added when token
    accounting landed, which is a breaking change to this signature.
    """
    dataset = Dataset(
        question=["q"], context=[["c"]], response=["r"]
    )
    results, usage = run_metrics(
        dataset,
        [response_relevancy],
        llm=FakeLLM(responses=["0.5"]),
        config=SERIAL,
    )
    assert results[0][0].score == 0.5
    assert usage.calls == 1


def test_progress_bar_path_runs() -> None:
    """The tqdm branch must not be broken by disuse."""
    dataset = Dataset(
        question=["q"], context=[["c"]], response=["r"]
    )
    result = evaluate(
        dataset,
        llm=FakeLLM(responses=["0.5"]),
        run_config=RunConfig(show_progress=True, max_workers=1),
    )
    assert result.scores == [[0.5]]


def test_dataset_with_progress_yields_nodes() -> None:
    """Dataset.with_progress still iterates data nodes."""
    dataset = Dataset(
        question=["a", "b"],
        context=[["c"], ["c"]],
        response=["r", "s"],
    )
    nodes = list(dataset.with_progress("Testing"))
    assert len(nodes) == 2
    assert all(isinstance(item, DataNode) for item in nodes)


# --------------------------- readers ---------------------------


def test_from_dataframe_round_trip() -> None:
    """A DataFrame in, a Dataset out."""
    frame = Dataset(
        question=["a", "b"],
        context=[["c1"], ["c2"]],
        response=["r", "s"],
    ).to_dataframe()
    dataset = from_dataframe(frame, return_as_dataset=True)
    assert len(dataset) == 2
    assert dataset[0].context == ["c1"]


def test_from_csv_round_trip(tmp_path: Path) -> None:
    """CSV survives the trip, including list-shaped context cells."""
    path = tmp_path / "data.csv"
    Dataset(
        question=["a", "b"],
        context=[["c1", "c2"], ["c3"]],
        response=["r", "s"],
    ).to_csv(path)

    dataset = from_csv(path)
    assert len(dataset) == 2
    assert dataset[0].context == ["c1", "c2"]
    assert dataset[1].context == ["c3"]


def test_from_csv_with_a_column_map(tmp_path: Path) -> None:
    """Renamed columns are read correctly."""
    path = tmp_path / "renamed.csv"
    path.write_text(
        "query,context,answer\nq1,\"['c1']\",a1\n", encoding="utf-8"
    )
    dataset = from_dataframe(
        __import__("pandas").read_csv(path),
        column_map=ColumnMap(question="query", response="answer"),
        return_as_dataset=True,
    )
    assert dataset[0].question == "q1"
    assert dataset[0].response == "a1"


def test_from_dict_returns_a_dataset_when_asked() -> None:
    """return_as_dataset promotes a single row."""
    result = from_dict(
        {"question": "q", "context": ["c"], "response": "r"},
        return_as_dataset=True,
    )
    assert isinstance(result, Dataset)
    assert len(result) == 1


# --------------------------- InstructConfig ---------------------------


def test_instruct_config_to_prompt() -> None:
    """The config converts to a usable prompt."""
    config = InstructConfig(
        metric_type=MetricType.BINARY,
        name="Politeness",
        instructions="Is it polite?",
        input_fields=["question", "response"],
    )
    prompt = config.to_prompt()
    assert prompt.name == "Politeness"
    assert prompt.input_keys == ["question", "response"]


def test_instruct_config_repr() -> None:
    """Readable in logs."""
    config = InstructConfig(
        metric_type=MetricType.BINARY,
        name="Politeness",
        instructions="...",
        input_fields=["response"],
    )
    assert "Politeness" in repr(config)


# --------------------------- names ---------------------------


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (ExactMatch(), "Exact Match"),
        (StringPresence(), "String Presence"),
        (LevenshteinRatio(), "Levenshtein Ratio"),
        (TokenF1(), "Token F1"),
        (RougeL(), "ROUGE-L"),
        (JsonValid(), "JSON Valid"),
        (HitRate(), "Hit Rate"),
        (MRR(), "MRR"),
        (PrecisionAtK(), "Precision"),
        (RecallAtK(), "Recall"),
        (MAP(), "MAP"),
        (NDCG(), "NDCG"),
    ],
)
def test_metric_names(metric: object, expected: str) -> None:
    """Names are what appear in reports, so pin them."""
    assert metric.name == expected


def test_names_are_unique_within_a_run() -> None:
    """Duplicate names would collide as DataFrame columns."""
    metrics = [
        ExactMatch(),
        TokenF1(),
        HitRate(),
        RecallAtK(k=5),
        RecallAtK(k=10),
    ]
    names = [item.name for item in metrics]
    assert len(names) == len(set(names))


def test_exact_match_singleton_matches_a_fresh_instance() -> None:
    """The module-level singletons are ordinary instances."""
    assert exact_match.name == ExactMatch().name
