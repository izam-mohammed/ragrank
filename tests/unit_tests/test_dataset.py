from typing import Dict, List, Union

import pytest
from pandas import DataFrame
from ragrank.bridge.pydantic import ValidationError
from ragrank.dataset import DataNode, Dataset, from_dict


@pytest.fixture
def valid_datanode() -> Dict[str, Union[str, List[str]]]:
    """Fixture for a valid data node."""
    return {
        "question": "What is the capital of France?",
        "context": ["France is a country located in Europe."],
        "response": "The capital of France is Paris.",
    }


@pytest.fixture
def invalid_datanode() -> Dict[str, Union[str, List[str]]]:
    """Fixture for an invalid data node with a non-string question."""
    return {
        "question": 123,  # Invalid question (not a string)
        "context": ["France is a country located in Europe."],
        "response": "The capital of France is Paris.",
    }


@pytest.fixture
def valid_dataset() -> Dict[str, Union[List[str], List[List[str]]]]:
    """Fixture for a valid dataset."""
    return {
        "question": ["Q1", "Q2"],
        "context": [["C1"], ["C2"]],
        "response": ["R1", "R2"],
    }


@pytest.fixture
def invalid_dataset() -> Dict[str, Union[List[str], List[List[str]]]]:
    """Fixture for an invalid dataset with a missing response."""
    return {
        "question": ["Q1", "Q2"],
        "context": [["C1"], ["C2"]],
        "response": ["R1"],  # Invalid, missing response for one question
    }


def test_valid_datanode(valid_datanode):
    """Test case for a valid data node."""
    data_node = DataNode(**valid_datanode)
    assert isinstance(data_node, DataNode), "Expected instance of DataNode."


def test_invalid_datanode(invalid_datanode):
    """Test case for an invalid data node."""
    with pytest.raises(Exception):
        DataNode(**invalid_datanode)


def test_valid_dataset(valid_dataset):
    """Test case for a valid dataset."""
    dataset = Dataset(**valid_dataset)
    assert isinstance(dataset, Dataset), "Expected instance of Dataset."


def test_invalid_dataset(invalid_dataset):
    """Test case for an invalid dataset."""
    with pytest.raises(ValueError):
        Dataset(**invalid_dataset)


def test_datanode_to_dataset(valid_datanode):
    """Test case for converting a data node to a dataset."""
    data_node = DataNode(**valid_datanode)
    dataset = data_node.to_dataset()
    assert isinstance(dataset, Dataset), "Expected instance of Dataset."


def test_dataset_len(valid_dataset):
    """Test case for getting the length of a dataset."""
    dataset = Dataset(**valid_dataset)
    assert len(dataset) == 2, "Expected length of dataset to be 2."


def test_dataset_subscription(valid_dataset):
    """Test case for accessing elements of a dataset."""
    dataset = Dataset(**valid_dataset)
    assert dataset[0].question == "Q1", "Expected first question to be 'Q1'."
    assert dataset[0].context == ["C1"], "Expected first context to be ['C1']."
    assert dataset[0].response == "R1", "Expected first response to be 'R1'."


def test_dataset_append(valid_datanode):
    """Test case for appending a data node to a dataset."""
    dataset = Dataset(question=["Q1"], context=[["C1"]], response=["R1"])
    data_node = DataNode(**valid_datanode)
    dataset.append(data_node)
    assert (
        len(dataset) == 2
    ), "Expected length of dataset to be 2 after appending."


def test_dataset_addition(valid_dataset):
    """Test case for adding two datasets."""
    dataset1 = Dataset(**valid_dataset)
    dataset2 = Dataset(question=["Q3"], context=[["C3"]], response=["R3"])
    combined_dataset = dataset1 + dataset2
    assert (
        len(combined_dataset) == 3
    ), "Expected length of combined dataset to be 3."


def test_dataset_to_pandas(valid_dataset):
    """Test case for converting a dataset to a pandas DataFrame."""
    dataset = Dataset(**valid_dataset)
    df = dataset.to_pandas()
    assert isinstance(df, DataFrame), "Expected instance of DataFrame."
    assert df.shape == (2, 3), "Expected DataFrame shape to be (2, 3)."


def test_from_dict_datanode(valid_datanode):
    """Test case for creating a data node from a dictionary."""
    data_node = from_dict(valid_datanode)
    assert isinstance(data_node, DataNode), "Expected instance of DataNode."


def test_from_dict_dataset(valid_dataset):
    """Test case for creating a dataset from a dictionary."""
    dataset = from_dict(valid_dataset)
    assert isinstance(dataset, Dataset), "Expected instance of Dataset."


def test_empty_input_data_node():
    """Test case for creating an empty data node."""
    with pytest.raises(ValidationError):
        DataNode()


def test_empty_input_dataset():
    """Test case for creating an empty dataset."""
    with pytest.raises(ValidationError):
        Dataset()
