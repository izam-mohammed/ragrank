"""Handle all of the things related to LLM in ragrank"""

from ragrank.llm.base import LLMConfig, LLMResult, default_llm
from ragrank.llm.BaseLLM import BaseLLM

__all__ = ["LLMConfig", "LLMResult", "BaseLLM", "default_llm"]
