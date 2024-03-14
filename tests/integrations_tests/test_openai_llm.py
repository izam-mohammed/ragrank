import pytest
from ragrank.integrations.llm import OpenaiLLM
from ragrank.llm import LLMResult


def test_llm_init():
    openai_llm = OpenaiLLM()
    assert openai_llm.name == "OpenAI LLM"


def test_generate_text_returns_llmresult():
    openai_llm = OpenaiLLM()
    result = openai_llm.generate_text("Test input")
    assert isinstance(result, LLMResult)


def test_generate_returns_llm_result():
    openai_llm = OpenaiLLM()
    results = openai_llm.generate(["How are you", "I am fine"])
    for result in results:
        assert isinstance(result, LLMResult)
    assert len(results) == 2
