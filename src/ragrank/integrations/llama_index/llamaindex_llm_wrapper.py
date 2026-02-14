"""Module for all of the llamaindex llm models"""

from __future__ import annotations

from time import time
from typing import Any, cast

try:
    from llama_index.core.base.llms.types import CompletionResponse
    from llama_index.core.llms.llm import (
        LLM as LlamaindexBaseLLM,  # noqa: N811
    )
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please install `llama-index` module by \n"
        "`pip install llama-index`"
    ) from ModuleNotFoundError

from ragrank.bridge.pydantic import (
    Field,
    field_validator,
)
from ragrank.llm import BaseLLM, LLMResult


class LlamaindexLLMWrapper(BaseLLM):
    """Wrapper class for Llamaindex Language Models.

    Attributes:
        llm (LLM): The Llamaindex Language Model.

    Properties:
        name (str): Get the name of the Llamaindex LLM Wrapper.
        llm_name (str): Get the name of the wrapped Llamaindex LLM.

    Methods:
        generate_text(): Generate text using the Llamaindex LLM.
    """

    llm: cast(LlamaindexBaseLLM, Any) = Field(  # type: ignore
        description="The Llamaindex Language Model."
    )

    @field_validator("llm")
    @classmethod
    def validator(
        cls: type[LlamaindexLLMWrapper], v: LlamaindexBaseLLM
    ) -> LlamaindexLLMWrapper:
        """
        Pydantic validation for Llamaindex LLM instance.
        Doing this because of the conflict in the v1 pydantic module.

        Raises:
            TypeError: If the type of the llamaindex llm is not correct.
        """
        if not isinstance(v, LlamaindexBaseLLM):
            raise TypeError(
                "the type of the Llamaindex llm is not valid\n\n"
                "Example:\n\nfrom llama_index.llms.openai import OpenAI\n"
                "### set your openai key as environment variable\n"
                "llm = LlamaindexLLMWrapper(llamaindex_llm=OpenAI())\n\n"
            ) from TypeError
        return v

    @property
    def name(self) -> str:
        """Get the name of the Llamaindex LLM Wrapper."""

        return f"Llamaindex LLM Wrapper for {self.llm_name}"

    @property
    def llm_name(self) -> str:
        """Get the name of the wrapped Llamaindex LLM."""

        return self.llm.metadata.model_name

    def generate_text(
        self,
        text: str,
    ) -> LLMResult:
        """Generate text using the Llamaindex LLM.

        Args:
            text (str): The input text.

        Returns:
            LLMResult: The generated text result.
        """

        start_time = time()
        llamaindex_result: CompletionResponse = self.llm.complete(
            text
        )
        message = llamaindex_result.text
        response_time = time() - start_time

        result = LLMResult(
            response=message,
            response_time=response_time,
            llm=self,
        )
        return result
