import pytest
from ragrank.integrations.llm import OpenaiLLM
from ragrank.llm import LLMResult


def test_llm_init():
    """Test initialization of OpenAI LLM."""
    openai_llm = OpenaiLLM()
    assert openai_llm.name == "OpenAI LLM", "mismatch in LLM name."


def test_generate_text_returns_llmresult():
    """Test if generate_text method returns an instance of LLMResult."""
    openai_llm = OpenaiLLM()
    result = openai_llm.generate_text("Test input")
    assert isinstance(result, LLMResult), "Expected an LLMResult instance."


def test_generate_returns_llm_result():
    """Test if generate method returns a list of LLMResult instances."""
    openai_llm = OpenaiLLM()
    results = openai_llm.generate(["How are you", "I am fine"])
    for result in results:
        assert isinstance(result, LLMResult), (
            f"generate method result :{result}\n"
            "Expected an LLMResult instance."
        )
    assert (
        len(results) == 2
    ), "generate method did not return the expected number of results."
