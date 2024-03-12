"""The base file for LLM module"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ragrank.bridge.pydantic import BaseModel
from ragrank.dataset import Dataset
from ragrank.prompt import Prompt


class BaseLLM(ABC):
    """Base LLM class for the llm in ragrank"""

    llm_config: LLMconfig

    def set_config(self, config: LLMconfig) -> None:
        """Setting the config for the base LLM"""
        self.llm_config = config

    @abstractmethod
    def generate_text(self, prompt: Prompt) -> LLMResult:
        """Result for single text input"""

    @abstractmethod
    def generate(self, dataset: Dataset) -> LLMResult:
        """generate response for dataset input"""


class LLMconfig(BaseModel):
    """Configuration for the input for llm"""

    temperature: float
    max_tokens: int
    seed: int
    top_p: int
    stop: List[str]


class LLMResult(BaseModel):
    """The result class for the llm"""

    response: str
    response_time: float
    llm: BaseLLM
    llm_config: LLMconfig
