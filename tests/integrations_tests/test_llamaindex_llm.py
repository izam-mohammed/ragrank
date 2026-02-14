"""Testing the LlamaindexLLMWrapper module"""

import pytest
from llama_index.llms.openai import OpenAI

pytestmark = pytest.mark.openai
from ragrank.integrations.llama_index import LlamaindexLLMWrapper
from ragrank.llm import LLMResult


def test_llm_init() -> None:
    """Test initialization of LlamaindexLLMWrapper."""

    openai_llm = OpenAI()
    llm = LlamaindexLLMWrapper(llm=openai_llm)
    assert (
        llm.name == "Llamaindex LLM Wrapper for gpt-3.5-turbo"
    ), "mismatch in LLM name."


def test_invalid_init() -> None:
    """Invalid initialization of LlamaindexLLMWrapper."""
    with pytest.raises(TypeError):
        LlamaindexLLMWrapper(llm="invalid llm")


def test_generate_text_returns_llmresult() -> None:
    """Test if generate_text method returns an instance of LLMResult."""

    openai_llm = OpenAI()
    llm = LlamaindexLLMWrapper(llm=openai_llm)
    result = llm.generate_text("Test input")
    assert isinstance(
        result, LLMResult
    ), "Expected an LLMResult instance."


def test_generate_returns_llm_result() -> None:
    """Test if generate method returns a list of LLMResult instances."""

    openai_llm = OpenAI()
    llm = LlamaindexLLMWrapper(llm=openai_llm)
    results = llm.generate(["How are you", "I am fine"])
    for result in results:
        assert isinstance(result, LLMResult), (
            f"generate method result :{result}\n"
            "Expected an LLMResult instance."
        )
    assert (
        len(results) == 2
    ), "generate method did not return the expected number of results."
