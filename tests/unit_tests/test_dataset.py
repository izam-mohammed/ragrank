"""Test cases for dataset module"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
import pytest
from ragrank.bridge.pydantic import ValidationError
from ragrank.dataset import (
    DataNode,
    Dataset,
    from_csv,
    from_dataframe,
    from_dict,
)
from ragrank.utils.common import eval_cell
from tqdm import tqdm


@pytest.fixture
def valid_datanode() -> Dict[str, str | List[str]]:
    """Fixture for a valid data node."""

    return {
        "question": "What is the capital of France?",
        "context": ["France is a country located in Europe."],
        "response": "The capital of France is Paris.",
    }


@pytest.fixture
def invalid_datanode() -> Dict[str, str | List[str]]:
    """Fixture for an invalid data node with a non-string question."""

    return {
        "question": 123,  # Invalid question (not a string)
        "context": ["France is a country located in Europe."],
        "response": "The capital of France is Paris.",
    }


@pytest.fixture
def valid_dataset() -> Dict[str, List[str] | List[List[str]]]:
    """Fixture for a valid dataset."""

    return {
        "question": ["Q1", "Q2"],
        "context": [["C1"], ["C2"]],
        "response": ["R1", "R2"],
    }


@pytest.fixture
def invalid_dataset() -> Dict[str, List[str] | List[List[str]]]:
    """Fixture for an invalid dataset with a missing response."""

    return {
        "question": ["Q1", "Q2"],
        "context": [["C1"], ["C2"]],
        "response": ["R1"],  # Invalid, missing response for one question
    }


def test_valid_datanode(valid_datanode: Dict[str, str | List[str]]) -> None:
    """Test case for a valid data node."""

    data_node = DataNode(**valid_datanode)
    assert isinstance(data_node, DataNode), "Expected instance of DataNode."


def test_invalid_datanode(
    invalid_datanode: Dict[str, str | List[str]],
) -> None:
    """Test case for an invalid data node."""

    with pytest.raises(ValidationError):
        DataNode(**invalid_datanode)


def test_valid_dataset(
    valid_dataset: Dict[str, List[str] | List[List[str]]],
) -> None:
    """Test case for a valid dataset."""

    dataset = Dataset(**valid_dataset)
    assert isinstance(dataset, Dataset), "Expected instance of Dataset."


def test_invalid_dataset(
    invalid_dataset: Dict[str, List[str] | List[List[str]]],
) -> None:
    """Test case for an invalid dataset."""

    with pytest.raises(ValueError):
        Dataset(**invalid_dataset)


def test_empty_input_data_node() -> None:
    """Test case for creating an empty data node."""

    with pytest.raises(ValidationError):
        DataNode()


def test_empty_input_dataset() -> None:
    """Test case for creating an empty dataset."""

    with pytest.raises(ValidationError):
        Dataset()


# datanode methods


def test_datanode_to_dataset(
    valid_datanode: Dict[str, str | List[str]],
) -> None:
    """Test case for converting a data node to a dataset."""

    data_node = DataNode(**valid_datanode)
    dataset = data_node.to_dataset()
    assert isinstance(dataset, Dataset), "Expected instance of Dataset."


# dataset methods


def test_dataset_len(
    valid_dataset: Dict[str, List[str] | List[List[str]]],
) -> None:
    """Test case for getting the length of a dataset."""

    dataset = Dataset(**valid_dataset)
    assert len(dataset) == 2, "Expected length of dataset to be 2."


def test_dataset_subscription(
    valid_dataset: Dict[str, List[str] | List[List[str]]],
) -> None:
    """Test case for accessing elements of a dataset."""

    dataset = Dataset(**valid_dataset)
    assert dataset[0].question == "Q1", "Expected first question to be 'Q1'."
    assert dataset[0].context == ["C1"], "Expected first context to be ['C1']."
    assert dataset[0].response == "R1", "Expected first response to be 'R1'."


def test_dataset_append(valid_datanode: Dict[str, str | List[str]]) -> None:
    """Test case for appending a data node to a dataset."""

    dataset = Dataset(question=["Q1"], context=[["C1"]], response=["R1"])
    data_node = DataNode(**valid_datanode)
    dataset.append(data_node)
    assert (
        len(dataset) == 2
    ), "Expected length of dataset to be 2 after appending."


def test_dataset_addition(
    valid_dataset: Dict[str, List[str] | List[List[str]]],
) -> None:
    """Test case for adding two datasets."""

    dataset1 = Dataset(**valid_dataset)
    dataset2 = Dataset(question=["Q3"], context=[["C3"]], response=["R3"])
    combined_dataset = dataset1 + dataset2
    assert (
        len(combined_dataset) == 3
    ), "Expected length of combined dataset to be 3."


def test_dataset_with_progress(
    valid_dataset: Dict[str, List[str] | List[List[str]]],
) -> None:
    """Test case for check the progress bar in the dataset iteration"""

    dataset = Dataset(**valid_dataset)
    dataset_with_progress = dataset.with_progress()
    assert isinstance(dataset_with_progress, tqdm), "Expected a tqdm object."


def test_dataset_to_dict(
    valid_dataset: Dict[str, List[str] | List[List[str]]],
) -> None:
    """Test case for check the to_dict function is working or not"""
    dataset = Dataset(**valid_dataset)
    dataset_dict = dataset.to_dict()
    assert isinstance(dataset_dict, dict), "Expected a dict object."
    assert (
        dataset_dict == valid_dataset
    ), "Mismatch in the expected dict object."


def test_dataset_to_dataframe(
    valid_dataset: Dict[str, List[str] | List[List[str]]],
) -> None:
    """Test case for converting a dataset to a pandas DataFrame."""

    dataset = Dataset(**valid_dataset)
    df = dataset.to_dataframe()
    assert isinstance(df, pd.DataFrame), "Expected instance of DataFrame."
    assert df.shape == (2, 3), "Expected DataFrame shape to be (2, 3)."


def test_dataset_to_csv(
    valid_dataset: Dict[str, List[str] | List[List[str]]],
) -> None:
    """Check the to_csv function with some data"""
    dataset = Dataset(**valid_dataset)
    filename = "testdata.csv"
    dataset.to_csv(path=filename)
    assert os.path.exists(
        filename
    ), "Expected a csv file 'testdata.csv' in the system."
    assert (
        (pd.read_csv(filename).map(eval_cell) == dataset.to_dataframe())
        .all()
        .all()
    ), "Expected content mismatch in the csv file."
    os.remove(filename)


# data reader


def test_from_dict_datanode(
    valid_datanode: Dict[str, str | List[str]],
) -> None:
    """Test case for creating a data node from a dictionary."""

    data_node = from_dict(valid_datanode)
    assert isinstance(data_node, DataNode), "Expected instance of DataNode."


def test_from_dict_dataset(
    valid_dataset: Dict[str, List[str] | List[List[str]]],
) -> None:
    """Test case for creating a dataset from a dictionary."""

    dataset = from_dict(valid_dataset)
    assert isinstance(dataset, Dataset), "Expected instance of Dataset."


@pytest.fixture
def valid_dataframe(
    valid_dataset: Dict[str, List[str] | List[List[str]]],
) -> pd.DataFrame:
    """Returns a valid dataframe"""

    return pd.DataFrame(valid_dataset)


def test_from_dataframe(
    valid_dataframe: pd.DataFrame,
) -> None:
    """check the from_df with valid dataset df"""

    dataset = from_dataframe(data=valid_dataframe)
    assert isinstance(dataset, Dataset), "Expected a Dataset instance."


def test_from_csv(
    valid_dataset: Dict[str, List[str] | List[List[str]]],
) -> None:
    """Check the from csv method"""

    data = from_dict(valid_dataset)
    filename = "test.csv"
    data.to_csv(filename)
    data_input = from_csv(filename)
    os.remove(filename)
    assert isinstance(data_input, Dataset), "Expected a Dataset instance."
    assert data == data_input, "Expected different content in the csv"
