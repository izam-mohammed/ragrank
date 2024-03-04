import pytest
from ragrank.prompt import Prompt


@pytest.fixture
def example_data():
    return [
        {"input": "How are you?", "output": "I'm fine, thank you!"},
        {"input": "What is your name?", "output": "My name is ChatGPT."},
    ]


@pytest.fixture
def valid_prompt_data(example_data):
    return {
        "name": "Example Prompt",
        "instructions": "Please answer the following questions.",
        "examples": example_data,
        "input_keys": ["input"],
        "output_key": "output",
    }


@pytest.fixture
def invalid_prompt_data():
    return {
        "name": "Invalid Prompt",
        "instructions": "This prompt has missing keys.",
        "input_keys": [],
        "output_key": "",
    }


def test_valid_prompt_creation(valid_prompt_data):
    """Test creation of a valid Prompt object."""

    prompt = Prompt(**valid_prompt_data)
    assert prompt.name == valid_prompt_data["name"]
    assert prompt.instructions == valid_prompt_data["instructions"]
    assert prompt.examples == valid_prompt_data["examples"]
    assert prompt.input_keys == valid_prompt_data["input_keys"]
    assert prompt.output_key == valid_prompt_data["output_key"]


def test_invalid_prompt_creation(invalid_prompt_data):
    """Test creation of an invalid Prompt object."""

    with pytest.raises(ValueError):
        prompt = Prompt(**invalid_prompt_data)


def test_validate_prompt_missing_examples():
    """Test validation for a prompt with missing examples."""

    invalid_prompt_data = {
        "name": "Missing Examples",
        "instructions": "This prompt has no examples.",
        "input_keys": ["input"],
        "output_key": "output",
    }
    with pytest.raises(ValueError):
        prompt = Prompt(**invalid_prompt_data)


def test_validate_prompt_missing_input_keys():
    """Test validation for a prompt with missing input keys."""

    invalid_prompt_data = {
        "name": "Missing Input Keys",
        "instructions": "This prompt has no input keys.",
        "examples": [{"input": "Some input", "output": "Some output"}],
        "output_key": "output",
    }
    with pytest.raises(ValueError):
        prompt = Prompt(**invalid_prompt_data)


def test_to_string(valid_prompt_data):
    """Test the to_string method of the Prompt class."""

    prompt = Prompt(**valid_prompt_data)
    prompt_str = prompt.to_string()
    assert isinstance(prompt_str, str)
    assert valid_prompt_data["name"] in prompt_str
    assert valid_prompt_data["instructions"] in prompt_str
    for example in valid_prompt_data["examples"]:
        assert example["input"] in prompt_str
        assert example["output"] in prompt_str
    for key in valid_prompt_data["input_keys"]:
        assert "{" + key + "}" in prompt_str
    assert valid_prompt_data["output_key"] in prompt_str


def test_get_examples(example_data):
    """Test the get_examples method of the Prompt class."""

    prompt = Prompt(
        name="Test Prompt",
        instructions="Test instructions",
        examples=example_data,
        input_keys=["input"],
        output_key="output",
    )
    assert prompt.get_examples() == example_data
    assert prompt.get_examples(1) == example_data[:1]
    assert prompt.get_examples(2) == example_data[:2]
    with pytest.raises(IndexError):
        prompt.get_examples(3)
