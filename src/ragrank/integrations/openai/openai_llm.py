"""Module for integrating OpenAI language model"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from ragrank.bridge.pydantic import Field, PrivateAttr
from ragrank.llm import BaseLLM, LLMResult
from ragrank.utils.llm import get_env_var

try:
    from openai import OpenAI
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'Please install `openai` with `pip install "ragrank[openai]"`'
    ) from None

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant"


class OpenaiLLM(BaseLLM):
    """Represents an OpenAI Language Model (LLM) for generating text.

    Attributes:
        model (str): The OpenAI model to use.
        system_message (str): The system message sent with each request.

    Methods:
        generate_text: Generates text using the OpenAI language model.
    """

    model: str = Field(
        default=DEFAULT_MODEL,
        description="The OpenAI model to use.",
    )
    system_message: str = Field(
        default=DEFAULT_SYSTEM_MESSAGE,
        repr=False,
        description="The system message sent with each request.",
    )

    _client: Any = PrivateAttr(default=None)

    @property
    def name(self) -> str:
        """
        Get the name of the language model.

        Returns:
            str: The name of the language model.
        """
        return f"OpenAI LLM ({self.model})"

    @property
    def client(self) -> OpenAI:
        """The OpenAI client, created once and reused.

        Building a client per call re-read the environment and re-opened
        a connection pool for every single row of an evaluation.

        Returns:
            OpenAI: The client.
        """
        if self._client is None:
            self._client = OpenAI(
                api_key=get_env_var("OPENAI_API_KEY")
            )
        return self._client

    def generate_text(self, text: str) -> LLMResult:
        """
        Generate text using the OpenAI language model.

        Args:
            text (str): The user's input text.

        Returns:
            LLMResult: Result containing the generated text and other metadata.
        """
        start_time = perf_counter()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": text},
            ],
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
            seed=self.llm_config.seed,
            top_p=self.llm_config.top_p,
            stop=self.llm_config.stop,
        )

        if not completion.choices:
            raise ValueError("Unable to generate the output")

        choice = completion.choices[0]
        usage = completion.usage

        return LLMResult(
            response=choice.message.content or "",
            response_time=perf_counter() - start_time,
            llm=self,
            llm_config=self.llm_config,
            finish_reason=choice.finish_reason,
            response_tokens=(
                usage.completion_tokens if usage else None
            ),
        )
