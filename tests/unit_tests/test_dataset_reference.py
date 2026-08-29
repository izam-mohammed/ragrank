"""Tests for the reference / retrieved_ids / reference_ids fields."""

from __future__ import annotations

import pytest
from ragrank.dataset import ColumnMap, DataNode, Dataset, from_dict


def test_optional_fields_default_to_none() -> None:
    """A node without ground truth is still valid."""
    node = DataNode(question="q", context=["c"], response="r")
    assert node.reference is None
    assert node.retrieved_ids is None
    assert node.reference_ids is None


def test_optional_fields_round_trip() -> None:
    """Ground truth survives node -> dataset -> node."""
    node = DataNode(
        question="q",
        context=["c"],
        response="r",
        reference="gt",
        retrieved_ids=["d1", "d2"],
        reference_ids=["d2"],
    )
    restored = node.to_dataset()[0]
    assert restored == node


def test_to_dict_omits_absent_optionals() -> None:
    """Legacy dicts must round-trip unchanged."""
    payload = {
        "question": ["a"],
        "context": [["c"]],
        "response": ["r"],
    }
    assert Dataset(**payload).to_dict() == payload


def test_to_dict_includes_present_optionals() -> None:
    """Supplied ground truth appears in the dict."""
    dataset = Dataset(
        question=["a"],
        context=[["c"]],
        response=["r"],
        reference=["gt"],
    )
    assert dataset.to_dict()["reference"] == ["gt"]


def test_validator_checks_optional_column_lengths() -> None:
    """A short reference column is caught at construction."""
    with pytest.raises(ValueError, match="reference"):
        Dataset(
            question=["a", "b"],
            context=[["c"], ["c"]],
            response=["r", "r"],
            reference=["only one"],
        )


def test_getitem_carries_optional_fields() -> None:
    """Indexing must not silently drop ground truth."""
    dataset = Dataset(
        question=["a", "b"],
        context=[["c"], ["c"]],
        response=["r", "s"],
        reference=["x", "y"],
        reference_ids=[["d1"], ["d2"]],
    )
    assert dataset[1].reference == "y"
    assert dataset[1].reference_ids == ["d2"]


def test_append_extends_optional_columns() -> None:
    """Appending keeps every column the same length."""
    dataset = Dataset(
        question=["a"],
        context=[["c"]],
        response=["r"],
        reference=["x"],
    )
    dataset.append(
        DataNode(
            question="b", context=["c"], response="s", reference="y"
        )
    )
    assert dataset.reference == ["x", "y"]
    assert len(dataset) == 2


def test_addition_combines_optional_columns() -> None:
    """Concatenation keeps ground truth aligned."""
    make = lambda q, ref: Dataset(  # noqa: E731
        question=[q],
        context=[["c"]],
        response=["r"],
        reference=[ref],
    )
    combined = make("a", "x") + make("b", "y")
    assert combined.reference == ["x", "y"]


def test_datanode_addition_still_returns_a_dataset() -> None:
    """DataNode + DataNode keeps working after the refactor."""
    left = DataNode(question="a", context=["c"], response="r")
    right = DataNode(question="b", context=["c"], response="s")
    combined = left + right
    assert isinstance(combined, Dataset)
    assert len(combined) == 2


def test_from_dict_reads_reference_when_present() -> None:
    """Optional columns are picked up when supplied."""
    node = from_dict({
        "question": "q",
        "context": ["c"],
        "response": "r",
        "reference": "gt",
    })
    assert node.reference == "gt"


def test_from_dict_ignores_absent_optional_columns() -> None:
    """Their absence is not an error."""
    node = from_dict({
        "question": "q",
        "context": ["c"],
        "response": "r",
    })
    assert node.reference is None


def test_from_dict_still_rejects_missing_required_columns() -> None:
    """A missing required column is still an error."""
    with pytest.raises(ValueError, match="response"):
        from_dict({"question": "q", "context": ["c"]})


def test_column_map_renames_optional_columns() -> None:
    """Ground truth can live under any column name."""
    node = from_dict(
        {
            "question": "q",
            "context": ["c"],
            "response": "r",
            "gt": "x",
        },
        column_map=ColumnMap(reference="gt"),
    )
    assert node.reference == "x"
