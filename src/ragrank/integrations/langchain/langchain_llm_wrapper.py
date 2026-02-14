"""Module for all of the langchain models in the langchain"""

from __future__ import annotations

from time import time
from typing import Any, cast

from ragrank.bridge.pydantic import Field, field_validator
from ragrank.llm import BaseLLM, LLMResult

try:
    from langchain_core.language_models import (
        BaseLanguageModel as LangchainBaseLanguageModel,
    )
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_core.outputs.llm_result import (
        LLMResult as LangchainLLMResult,
    )
    from langchain_core.prompt_values import PromptValue
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please install `langchain-core` module by \n"
        "`pip install langchain-core`"
    ) from ModuleNotFoundError


class LangchainLLMWrapper(BaseLLM):
    """Wrapper class for Langchain Language Models.

    Attributes:
        llm (LangchainBaseLLM): The Langchain Language Model.

    Properties:
        name (str): Get the name of the Langchain LLM Wrapper.
        llm_name (str): Get the name of the wrapped Langchain LLM.

    Methods:
        generate_text(): Generate text using the Langchain LLM.
    """

    llm: cast(LangchainBaseLanguageModel, Any) = Field(  # type: ignore
        description="The Langchain Language Model."
    )

    @field_validator("llm")
    @classmethod
    def validator(
        cls: type[LangchainLLMWrapper], v: LangchainBaseLanguageModel
    ) -> LangchainBaseLanguageModel:
        """Validating the langchain base language seperately
        Doing this because of the conflict in the v1 pydantic module.

        Raises:
            TypeError: If the type of the langchain llm is not valid
        """

        if not isinstance(v, LangchainBaseLanguageModel):
            raise TypeError(
                "the type of the langchain llm is not valid\n\n"
                "Example:\n\nfrom langchain_openai.llms import OpenAI\n"
                "### set your openai key as environment variable\n"
                "llm = LangchainLLMWrapper(langchain_llm=OpenAI())\n\n"
            ) from TypeError
        return v

    @property
    def name(self) -> str:
        """Get the name of the Langchain LLM Wrapper."""

        return f"Langchain LLM Wrapper for {self.llm_name}"

    @property
    def llm_name(self) -> str:
        """Get the name of the wrapped Langchain LLM."""

        return self.llm.get_name()

    def generate_text(
        self,
        text: str,
    ) -> LLMResult:
        """Generate text using the Langchain LLM.

        Args:
            text (str): The input text.

        Returns:
            LLMResult: The generated text result.
        """

        start_time = time()
        prompt = RagrankPromptValue(prompt_str=text)
        langchain_result: LangchainLLMResult = (
            self.llm.generate_prompt(prompts=[prompt])
        )
        message = langchain_result.generations[0][0].text
        response_tokens = langchain_result.llm_output["token_usage"][
            "completion_tokens"
        ]
        response_time = time() - start_time

        result = LLMResult(
            response=message,
            response_time=response_time,
            response_tokens=response_tokens,
            llm=self,
            llm_config=self.llm_config,
        )
        return result


class RagrankPromptValue(PromptValue):
    """Wrapper class for prompts used with Ragrank.

    Attributes:
        prompt_str (str): The prompt string.
    """

    prompt_str: str

    def to_messages(self) -> list[BaseMessage]:
        """Convert the prompt to a list of BaseMessage objects."""

        return [HumanMessage(content=self.to_string())]

    def to_string(self) -> str:
        """Convert the prompt to a string."""

        return self.prompt_str
