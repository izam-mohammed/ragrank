"""The flat JSON interchange format, and metric cost tiers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragrank.dataset import (
    ColumnMap,
    Dataset,
    from_json,
    from_jsonl,
    from_records,
)
from ragrank.metric import (
    CostTier,
    context_relevancy,
    exact_match,
    faithfulness,
    hit_rate,
    response_relevancy,
    semantic_similarity,
)
from ragrank.metric._custom.jury import Jury

RECORDS = [
    {
        "question": "who wrote it",
        "context": ["Ada wrote it."],
        "response": "Ada",
    },
    {
        "question": "when",
        "context": ["It was 1843."],
        "response": "1843",
    },
]


def test_records_round_trip() -> None:
    dataset = from_records(RECORDS)
    assert len(dataset) == 2
    assert dataset.to_records() == RECORDS


def test_records_carry_optional_columns() -> None:
    records = [
        {**item, "reference": "ground truth"} for item in RECORDS
    ]
    dataset = from_records(records)
    assert dataset.reference == ["ground truth", "ground truth"]
    assert dataset.to_records() == records


def test_a_string_context_is_wrapped_into_one_chunk() -> None:
    dataset = from_records([
        {
            "question": "q",
            "context": "a single chunk",
            "response": "r",
        }
    ])
    assert dataset.context == [["a single chunk"]]


def test_unknown_record_fields_are_dropped() -> None:
    dataset = from_records([
        {**item, "trace_id": "abc"} for item in RECORDS
    ])
    assert dataset.to_records() == RECORDS


def test_records_reject_ragged_fields() -> None:
    with pytest.raises(ValueError, match="Every record must"):
        from_records([
            RECORDS[0],
            {"question": "q", "context": ["c"]},
        ])


def test_records_reject_non_mappings() -> None:
    with pytest.raises(ValueError, match="not a mapping"):
        from_records([RECORDS[0], "nope"])


def test_records_reject_an_empty_list() -> None:
    with pytest.raises(ValueError, match="No records"):
        from_records([])


def test_from_json_reads_a_literal_document() -> None:
    dataset = from_json(json.dumps(RECORDS))
    assert dataset.to_records() == RECORDS


def test_from_json_reads_parsed_records() -> None:
    assert from_json(RECORDS).to_records() == RECORDS


def test_from_json_reads_a_file(tmp_path: Path) -> None:
    path = tmp_path / "outputs.json"
    path.write_text(json.dumps(RECORDS), encoding="utf-8")

    assert from_json(path).to_records() == RECORDS
    assert from_json(str(path)).to_records() == RECORDS


def test_from_json_reads_the_column_oriented_shape() -> None:
    columns = {
        "question": ["a", "b"],
        "context": [["x"], ["y"]],
        "response": ["1", "2"],
    }
    dataset = from_json(json.dumps(columns))
    assert isinstance(dataset, Dataset)
    assert dataset.question == ["a", "b"]


def test_from_json_rejects_a_bare_scalar(tmp_path: Path) -> None:
    path = tmp_path / "scalar.json"
    path.write_text("12", encoding="utf-8")
    with pytest.raises(ValueError, match="list of records"):
        from_json(path)


def test_a_non_json_looking_string_is_treated_as_a_path() -> None:
    with pytest.raises(FileNotFoundError):
        from_json("no/such/file.json")


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "outputs.jsonl"
    from_records(RECORDS).to_jsonl(path)

    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert from_jsonl(path).to_records() == RECORDS


def test_jsonl_skips_blank_lines() -> None:
    text = "\n".join(json.dumps(item) for item in RECORDS) + "\n\n"
    assert from_jsonl(text).to_records() == RECORDS


def test_jsonl_names_the_line_it_could_not_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(RECORDS[0]) + "\nnot json\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Line 2"):
        from_jsonl(path)


def test_to_json_writes_and_returns(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    text = from_records(RECORDS).to_json(path, indent=2)

    assert json.loads(text) == RECORDS
    assert json.loads(path.read_text(encoding="utf-8")) == RECORDS


def test_records_respect_a_column_map() -> None:
    records = [
        {"query": "q", "chunks": ["c"], "answer": "a"},
    ]
    dataset = from_records(
        records,
        column_map=ColumnMap(
            question="query", context="chunks", response="answer"
        ),
    )
    assert dataset.question == ["q"]
    assert dataset.response == ["a"]


@pytest.mark.parametrize(
    ("metric", "tier"),
    [
        (exact_match, CostTier.FREE),
        (hit_rate, CostTier.FREE),
        (semantic_similarity, CostTier.EMBEDDING),
        (response_relevancy, CostTier.LLM),
        (context_relevancy, CostTier.LLM_HEAVY),
        (faithfulness, CostTier.LLM_HEAVY),
    ],
)
def test_metrics_declare_a_cost_tier(metric, tier) -> None:
    assert metric.cost_tier is tier


def test_a_jury_is_always_the_heavy_tier() -> None:
    jury = Jury(judges=[response_relevancy])
    assert jury.cost_tier is CostTier.LLM_HEAVY


def test_every_metric_declares_a_tier() -> None:
    import ragrank.metric as registry

    from ragrank.metric import BaseMetric

    metrics = [
        getattr(registry, name)
        for name in registry.__all__
        if isinstance(getattr(registry, name, None), BaseMetric)
    ]
    assert metrics
    assert all(
        isinstance(item.cost_tier, CostTier) for item in metrics
    )
