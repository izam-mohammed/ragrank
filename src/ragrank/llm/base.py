"""Base for the llm module"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from ragrank.bridge.pydantic import BaseModel


class BaseLLM(BaseModel, ABC):
    """Base LLM class for the LLM in ragrank"""

    def set_config(self, config: LLMConfig) -> None:
        """
        Set the configuration for the base LLM.

        Args:
            config (LLMConfig): The configuration for the LLM.
        """
        self.llm_config = config

    @abstractmethod
    def generate_text(self, text: str) -> LLMResult:
        """
        Generate the result for a single text input.

        Args:
            text (str): The input text.

        Returns:
            LLMResult: The result of the LLM generation.
        """

    def generate(self, texts: Sequence[str]) -> List[LLMResult]:
        """
        Generate responses for a dataset input.

        Args:
            texts (Sequence[str]): A sequence of input texts.

        Returns:
            List[LLMResult]: A list of LLM results.
        """
        return [self.generate_text(text) for text in texts]


class LLMConfig(BaseModel):
    """Configuration for the input for LLM."""

    temperature: float
    max_tokens: int
    seed: int
    top_p: float
    stop: List[str]


class LLMResult(BaseModel):
    """The result class for the LLM."""

    class Config:
        arbitrary_types_allowed = True

    response: str
    response_time: Optional[float] = None
    llm: Optional[BaseLLM] = None
    llm_config: Optional[LLMConfig] = None


def default_llm() -> BaseLLM:
    """The default llm module"""
    from ragrank.integrations.llm import OpenaiLLM

    return OpenaiLLM
