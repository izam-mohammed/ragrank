"""Testing the langchain llm wrapper module with openai"""

import pytest
from langchain_openai.llms import OpenAI

pytestmark = pytest.mark.openai
from ragrank.integrations.langchain import LangchainLLMWrapper
from ragrank.llm import LLMResult


def test_llm_init() -> None:
    """Test initialization of LangchainLLMWrapper."""

    openai_llm = OpenAI()
    llm = LangchainLLMWrapper(llm=openai_llm)
    assert (
        llm.name == "Langchain LLM Wrapper for OpenAI"
    ), "mismatch in LLM name."


def test_invalid_init() -> None:
    """Invalid initialization of LangchainLLMWrapper."""
    with pytest.raises(TypeError):
        LangchainLLMWrapper(llm="invalid llm")


def test_generate_text_returns_llmresult() -> None:
    """Test if generate_text method returns an instance of LLMResult."""

    openai_llm = OpenAI()
    llm = LangchainLLMWrapper(llm=openai_llm)
    result = llm.generate_text("Test input")
    assert isinstance(
        result, LLMResult
    ), "Expected an LLMResult instance."


def test_generate_returns_llm_result() -> None:
    """Test if generate method returns a list of LLMResult instances."""

    openai_llm = OpenAI()
    llm = LangchainLLMWrapper(llm=openai_llm)
    results = llm.generate(["How are you", "I am fine"])
    for result in results:
        assert isinstance(result, LLMResult), (
            f"generate method result :{result}\n"
            "Expected an LLMResult instance."
        )
    assert (
        len(results) == 2
    ), "generate method did not return the expected number of results."
