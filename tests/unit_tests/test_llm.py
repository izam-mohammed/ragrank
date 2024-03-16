import pytest
from ragrank.llm import BaseLLM, LLMConfig, LLMResult, default_llm


@pytest.fixture
def base_llm():
    """Fixture to create an instance of BaseLLM."""
    BaseLLM.__abstractmethods__ = set()
    return BaseLLM()


@pytest.fixture
def llm_config_valid_dict():
    """Fixture to return a valid LLMConfig dictionary."""
    return {
        "temperature": 0.8,
        "max_tokens": 200,
        "seed": 123,
        "top_p": 0.9,
        "stop": ["</s>", "<|endoftext|>"],
    }


@pytest.fixture
def llm_config_invalid_dict():
    """Fixture to return an invalid LLMConfig dictionary
    with incorrect values."""
    return {
        "temperature": -0.5,
        "max_tokens": 500,
        "seed": 999,
        "top_p": 2.0,
        "stop": ["</s>", "<|endoftext|>"],
    }


@pytest.fixture
def llm_config_invalid_type_dict():
    """Fixture to return an invalid LLMConfig dictionary
    with incorrect data types."""
    return {
        "temperature": True,
        "max_tokens": "invalid",
        "seed": None,
        "top_p": "invalid",
        "stop": ["</s>", "<|endoftext|>"],
    }


@pytest.fixture
def llm_result_valid_dict():
    """Fixture to return a valid LLMResult dictionary."""
    return {
        "response": "This is a valid response.",
        "response_time": 0.5,
        "finish_reason": "Complete",
        "response_tokens": 50,
        "llm": default_llm(),
        "llm_config": LLMConfig(),
    }


@pytest.fixture
def llm_result_invalid_dict():
    """Fixture to return an invalid LLMResult dictionary
    with incorrect values or types."""
    return {
        "response": 123,
        "response_time": "invalid",
        "finish_reason": 456,
        "response_tokens": "invalid",
        "llm": "invalid",
        "llm_config": "invalid",
    }


def test_llm_config_initialization_valid(llm_config_valid_dict):
    """Test case to validate the initialization of LLMConfig
    with valid parameters."""
    config = LLMConfig(**llm_config_valid_dict)
    assert config.temperature == 0.8, "Expected temperature to be 0.8"
    assert config.max_tokens == 200, "Expected max tokens to be 200"
    assert config.seed == 123, "Expected seed to be 123"
    assert config.top_p == 0.9, "Expected top p to be 0.9"
    assert config.stop == ["</s>", "<|endoftext|>"], "Stop assertion failed"


def test_llm_config_initialization_invalid(
    llm_config_invalid_dict, llm_config_invalid_type_dict
):
    """Test case to validate the initialization of LLMConfig
    with invalid parameters."""
    with pytest.raises(ValueError):
        LLMConfig(**llm_config_invalid_dict)

    with pytest.raises(ValueError):
        LLMConfig(**llm_config_invalid_type_dict)


def test_llm_config_default_check():
    """Test case to check the default values of LLMConfig."""
    config = LLMConfig()
    assert config.temperature == 1.0, "Expected default temperature to be 1.0"
    assert config.max_tokens == 300, "Expected default max tokens to be 300"
    assert config.seed == 44, "Expected default seed to be 44"
    assert config.top_p == 1.0, "Expected default top p to be 1.0"
    assert config.stop is None, "Expected default stop tokens to be None"


def test_llm_result_initialization_valid(llm_result_valid_dict):
    """Test case to validate the initialization of LLMResult
    with valid parameters."""
    result = LLMResult(**llm_result_valid_dict)
    assert (
        result.response == "This is a valid response."
    ), "Expected response to match"
    assert result.response_time == 0.5, "Expected response time to be 0.5"
    assert (
        result.finish_reason == "Complete"
    ), "Expected finish reason to be 'Complete'"
    assert result.response_tokens == 50, "Expected response tokens to be 50"
    assert isinstance(result.llm, BaseLLM), "Expected an instance of BaseLLM"
    assert isinstance(
        result.llm_config, LLMConfig
    ), "Expected an instance of LLMConfig"


def test_llm_result_initialization_invalid(llm_result_invalid_dict):
    """Test case to validate the initialization of LLMResult
    with invalid parameters."""
    with pytest.raises(ValueError):
        LLMResult(**llm_result_invalid_dict)


def test_llm_result_default_check():
    """Test case to check the default values of LLMResult."""
    result = LLMResult(response="Default response")
    assert result.response == "Default response", "Expected response to match"
    assert result.response_time is None, "Expected response time to be None"
    assert result.finish_reason is None, "Expected finish reason to be None"
    assert (
        result.response_tokens is None
    ), "Expected response tokens to be None"
    assert result.llm is None, "Expected llm to be None"
    assert result.llm_config is None, "Expected llm_config to be None"


def test_base_llm_set_config(base_llm, llm_config_valid_dict):
    """Test case to set configuration for BaseLLM."""
    config = LLMConfig(**llm_config_valid_dict)
    base_llm.set_config(config)
    assert (
        base_llm.llm_config == config
    ), "Expected llm_config to match the provided configuration"
