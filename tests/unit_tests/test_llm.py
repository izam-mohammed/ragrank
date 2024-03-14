import pytest
from ragrank.llm import BaseLLM, LLMConfig, LLMResult, default_llm


@pytest.fixture
def base_llm():
    BaseLLM.__abstractmethods__ = set()
    return BaseLLM()


@pytest.fixture
def llm_config_valid_dict():
    return {
        "temperature": 0.8,
        "max_tokens": 200,
        "seed": 123,
        "top_p": 0.9,
        "stop": ["</s>", "<|endoftext|>"],
    }


@pytest.fixture
def llm_config_invalid_dict():
    return {
        "temperature": -0.5,
        "max_tokens": 500,
        "seed": 999,
        "top_p": 2.0,
        "stop": ["</s>", "<|endoftext|>"],
    }


@pytest.fixture
def llm_config_invalid_type_dict():
    return {
        "temperature": True,
        "max_tokens": "invalid",
        "seed": None,
        "top_p": "invalid",
        "stop": ["</s>", "<|endoftext|>"],
    }


@pytest.fixture
def llm_result_valid_dict():
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
    return {
        "response": 123,
        "response_time": "invalid",
        "finish_reason": 456,
        "response_tokens": "invalid",
        "llm": "invalid",
        "llm_config": "invalid",
    }


def test_llm_config_initialization_valid(llm_config_valid_dict):
    config = LLMConfig(**llm_config_valid_dict)
    assert config.temperature == 0.8
    assert config.max_tokens == 200
    assert config.seed == 123
    assert config.top_p == 0.9
    assert config.stop == ["</s>", "<|endoftext|>"]


def test_llm_config_initialization_invalid(
    llm_config_invalid_dict, llm_config_invalid_type_dict
):
    with pytest.raises(ValueError):
        LLMConfig(**llm_config_invalid_dict)

    with pytest.raises(ValueError):
        LLMConfig(**llm_config_invalid_type_dict)


def test_llm_config_default_check():
    config = LLMConfig()
    assert config.temperature == 1.0
    assert config.max_tokens == 300
    assert config.seed == 44
    assert config.top_p == 1.0
    assert config.stop is None


def test_llm_result_initialization_valid(llm_result_valid_dict):
    result = LLMResult(**llm_result_valid_dict)
    assert result.response == "This is a valid response."
    assert result.response_time == 0.5
    assert result.finish_reason == "Complete"
    assert result.response_tokens == 50
    assert isinstance(result.llm, BaseLLM)
    assert isinstance(result.llm_config, LLMConfig)


def test_llm_result_initialization_invalid(llm_result_invalid_dict):
    with pytest.raises(ValueError):
        LLMResult(**llm_result_invalid_dict)


def test_llm_result_default_check():
    result = LLMResult(response="Default response")
    assert result.response == "Default response"
    assert result.response_time is None
    assert result.finish_reason is None
    assert result.response_tokens is None
    assert result.llm is None
    assert result.llm_config is None


def test_base_llm_set_config(base_llm, llm_config_valid_dict):
    config = LLMConfig(**llm_config_valid_dict)
    base_llm.set_config(config)
    assert base_llm.llm_config == config
