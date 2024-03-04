import pytest
from ragrank.dataset import Dataset, from_dict
from ragrank.exceptions import ValidationError


@pytest.fixture
def valid_dict_data():
    """Fixture for valid dictionary data."""
    return {
        "question": "What is the capital of France?",
        "context": [
            "Paris is the capital of France.",
            "It is located in Europe.",
        ],
        "response": "The capital of France is Paris.",
    }


@pytest.fixture
def invalid_dict_data():
    """Fixture for invalid dictionary data."""
    return {
        "question": "What is the capital of France?",
        "context": "Paris is the capital of France.",
        "response": "The capital of France is Paris.",
    }


def test_from_dict_valid(valid_dict_data):
    """Test creation of a Dataset object from a valid dictionary."""

    dataset = from_dict(valid_dict_data)
    assert isinstance(dataset, Dataset)
    assert dataset.question == valid_dict_data["question"]
    assert dataset.context == valid_dict_data["context"]
    assert dataset.response == valid_dict_data["response"]


def test_from_dict_invalid(invalid_dict_data):
    """Test creation of a Dataset object from an invalid dictionary."""

    with pytest.raises(ValidationError):
        data = from_dict(invalid_dict_data)


def test_dataset_creation_valid(valid_dict_data):
    """Test creation of a Dataset object with valid data."""
    dataset = Dataset(**valid_dict_data)
    assert isinstance(dataset, Dataset)
    assert dataset.question == valid_dict_data["question"]
    assert dataset.context == valid_dict_data["context"]
    assert dataset.response == valid_dict_data["response"]


def test_dataset_creation_invalid_context(invalid_dict_data):
    """Test creation of a Dataset object with invalid context."""
    with pytest.raises(ValueError):
        Dataset(**invalid_dict_data)
