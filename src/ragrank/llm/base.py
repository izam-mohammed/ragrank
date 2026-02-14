"""Base for the llm module"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ragrank.bridge.pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    validate_call,
)


class LLMConfig(BaseModel):
    """
    Configuration settings for Language Model (LLM).

    Attributes:
        temperature (float): Sampling temperature for text generation.
            Default is 1.0.
        max_tokens (int): Maximum number of tokens to generate.
            Default is 300.
        seed (int): Random seed for text generation. Default is 44.
        top_p (float): Sampling top probability for text generation.
            Default is 1.0.
        stop (List[str] | None): List of tokens at which text
            generation should stop.
    """

    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for text generation.",
    )
    max_tokens: int = Field(
        default=300,
        description="Maximum number of tokens to generate.",
    )
    seed: int = Field(
        default=44, description="Random seed for text generation."
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Sampling top probability for text generation.",
    )
    stop: list[str] | None = Field(
        default=None,
        description="List of tokens at which text generation should stop",
    )


class BaseLLM(BaseModel, ABC):
    """
    Abstract base class for Language Model (LLM).

    This class provides an interface for interacting with language models.

    Attributes:
        llm_config (LLMConfig): Configuration settings for the LLM.

    Methods:
        set_config: Set configuration settings for the LLM.
        generate_text: Generate text based on input text.
        generate: Generate responses for a sequence of input texts.
    """

    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True
    )

    llm_config: LLMConfig = Field(
        repr=False, default_factory=LLMConfig
    )

    @validate_call
    def set_config(self, config: LLMConfig) -> None:
        """
        Set the configuration for the base LLM.

        Args:
            config (LLMConfig): The configuration for the LLM.
        """
        self.llm_config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the Language Model.

        Returns:
            str: Name of the Language Model.
        """
        return "Base LLM"

    @abstractmethod
    def generate_text(self, text: str) -> LLMResult:
        """
        Generate the result for a single text input.

        Args:
            text (str): The input text.

        Returns:
            LLMResult: The result of the LLM generation.
        """

    def generate(self, texts: Sequence[str]) -> list[LLMResult]:
        """
        Generate responses for a dataset input.

        Args:
            texts (Sequence[str]): A sequence of input texts.

        Returns:
            List[LLMResult]: A List of LLM results.
        """
        return [self.generate_text(text) for text in texts]

    def __repr__(self) -> str:
        """
        Return a string representation of the LLM.

        Returns:
            str: String representation of the LLM.
        """

        return self.name


class LLMResult(BaseModel):
    """
    Result of Language Model (LLM) generation.

    Attributes:
        response (str): Generated text response.
        response_time (float | None): Time taken for text generation.
        finish_reason (str | None): Reason for completion of
            text generation.
        response_tokens (int | None): Number of tokens in the
            generated response.
        llm (BaseLLM | None): Instance of the LLM used for generation.
        llm_config (LLMConfig | None): Configuration settings
            used for generation.
    """

    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True, frozen=True
    )

    response: str = Field(description="Generated text response.")
    response_time: float | None = Field(
        default=None, description="Time taken for text generation."
    )
    finish_reason: str | None = Field(
        default=None,
        description="Reason for completion of text generation",
    )
    response_tokens: int | None = Field(
        default=None,
        description="Number of tokens in the generated response.",
    )
    llm: BaseLLM | None = Field(
        default=None,
        description="Instance of the LLM used for generation.",
    )
    llm_config: LLMConfig | None = Field(
        default=None,
        description="Configuration settings used for generation.",
    )


def default_llm() -> BaseLLM:
    """
    Get the default Language Model (LLM) instance.

    Returns:
        BaseLLM: Default LLM instance.
    """
    from ragrank.integrations.openai import OpenaiLLM

    llm = OpenaiLLM()
    return llm
