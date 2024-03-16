from typing import List

import pytest
from ragrank.prompt import Prompt


@pytest.fixture
def valid_prompt_dict():
    """Fixture to provide a valid prompt dictionary."""
    return {
        "name": "Valid Prompt",
        "instructions": "Valid instructions",
        "examples": [{"input": "valid input", "output": "valid output"}],
        "input_keys": ["input"],
        "output_key": "output",
    }


@pytest.fixture
def invalid_prompt_empty_input_keys_dict():
    """Fixture to provide a prompt dictionary with empty input keys."""
    return {
        "name": "Invalid Prompt",
        "instructions": "Invalid instructions",
        "examples": [{"input": "invalid input", "output": "invalid output"}],
        "input_keys": [],
        "output_key": "output",
    }


@pytest.fixture
def invalid_prompt_mismatched_keys_dict():
    """Fixture to provide a prompt dictionary with mismatched keys."""
    return {
        "name": "Mismatched Keys Prompt",
        "instructions": "Mismatched keys instructions",
        "examples": [{"input_1": "input 1", "output": "output 1"}],
        "input_keys": ["input_2"],
        "output_key": "output",
    }


@pytest.fixture
def example_prompt_dict():
    """Fixture to provide an example prompt dictionary."""
    return {
        "name": "Test Prompt",
        "instructions": "Testing instructions",
        "examples": [
            {"input": "example input 1", "output": "example output 1"},
            {"input": "example input 2", "output": "example output 2"},
        ],
        "input_keys": ["input"],
        "output_key": "output",
    }


def test_validate_prompt(
    valid_prompt_dict,
    invalid_prompt_empty_input_keys_dict,
    invalid_prompt_mismatched_keys_dict,
):
    """Test validation of prompts."""
    valid_prompt = Prompt(**valid_prompt_dict)

    with pytest.raises(ValueError):
        invalid_prompt_empty_input_keys = Prompt(
            **invalid_prompt_empty_input_keys_dict
        )

    with pytest.raises(ValueError):
        invalid_prompt_mismatched_keys = Prompt(
            **invalid_prompt_mismatched_keys_dict
        )


def test_to_string(example_prompt_dict):
    """Test conversion of prompt to string."""
    example_prompt = Prompt(**example_prompt_dict)
    expected_output = (
        "Test Prompt\n\nTesting instructions\n\n"
        "input: example input 1\noutput: example output 1\n\n"
        "input: example input 2\noutput: example output 2\n\n"
        "input: {input}\n"
        "output:\n"
    )
    assert (
        example_prompt.to_string() == expected_output
    ), "Incorrect string representation"


def test_get_examples(example_prompt_dict):
    """Test retrieval of examples from prompt."""
    example_prompt = Prompt(**example_prompt_dict)

    assert (
        example_prompt.get_examples() == example_prompt.examples
    ), "Examples mismatch"
    assert (
        example_prompt.get_examples(1) == example_prompt.examples[:1]
    ), "Limited examples mismatch"
    with pytest.raises(IndexError):
        example_prompt.get_examples(5)
