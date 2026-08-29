"""One judge that speaks to every provider.

ragrank shipped a hand-written wrapper per provider, which meant the
library only really judged with OpenAI: anyone on Claude, Gemini or a
local Ollama had to route through the LangChain adapter to get here.

LiteLLM already normalises a hundred-odd providers onto one call, so
this is a single wrapper rather than a growing directory of them. Model
strings are LiteLLM's own -- ``anthropic/claude-sonnet-4-5``,
``gemini/gemini-2.0-flash``, ``ollama/llama3`` -- and the credential
lives wherever that provider expects it.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from ragrank.bridge.pydantic import Field
from ragrank.llm import BaseLLM, LLMResult
from ragrank.utils.optional import require

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant"


class LiteLLM(BaseLLM):
    """A judge backed by LiteLLM, so any supported provider works.

    Attributes:
        model (str): A LiteLLM model string, provider-prefixed for
            anything other than OpenAI.
        system_message (str): The system message sent with each request.
        api_key (str | None): Passed through when set. Leaving it None
            lets LiteLLM read the provider's usual environment variable,
            which is normally what you want.
        api_base (str | None): Override the endpoint, for a self-hosted
            or proxied model.
        extra_params (dict[str, Any]): Anything else to forward to
            `litellm.completion` -- an API version, a custom provider,
            a routing header.

    Examples::

        from ragrank import evaluate
        from ragrank.integrations.litellm import LiteLLM

        judge = LiteLLM(model="anthropic/claude-sonnet-4-5")
        result = evaluate(dataset, llm=judge)
    """

    model: str = Field(
        default=DEFAULT_MODEL,
        description="A LiteLLM model string.",
    )
    system_message: str = Field(
        default=DEFAULT_SYSTEM_MESSAGE,
        repr=False,
        description="The system message sent with each request.",
    )
    api_key: str | None = Field(
        default=None,
        repr=False,
        description="Provider credential, if not read from the env.",
    )
    api_base: str | None = Field(
        default=None,
        repr=False,
        description="Endpoint override, for self-hosted models.",
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        repr=False,
        description="Extra keyword arguments for litellm.completion.",
    )

    @property
    def name(self) -> str:
        """Get the name of the language model.

        Returns:
            str: The name of the language model.
        """
        return f"LiteLLM ({self.model})"

    def _completion_kwargs(self, text: str) -> dict[str, Any]:
        """Build the arguments for one completion call.

        Only options the caller actually set are forwarded. Providers
        differ in what they accept, and sending `seed` or `stop` as an
        explicit None is rejected by some of them.

        Args:
            text (str): The user's input text.

        Returns:
            dict[str, Any]: Keyword arguments for `litellm.completion`.
        """
        config = self.llm_config
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": text},
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        }
        if config.stop is not None:
            kwargs["stop"] = config.stop
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        kwargs.update(self.extra_params)
        return kwargs

    def generate_text(self, text: str) -> LLMResult:
        """Generate text through LiteLLM.

        Args:
            text (str): The user's input text.

        Returns:
            LLMResult: The generated text and its metadata.

        Raises:
            ValueError: If the provider returned no choices.
        """
        litellm = require("litellm", "litellm")

        start_time = perf_counter()
        completion = litellm.completion(
            **self._completion_kwargs(text)
        )

        choices = getattr(completion, "choices", None)
        if not choices:
            raise ValueError(
                f"{self.name} returned no choices for the prompt."
            )

        choice = choices[0]
        usage = getattr(completion, "usage", None)

        return LLMResult(
            response=choice.message.content or "",
            response_time=perf_counter() - start_time,
            llm=self,
            llm_config=self.llm_config,
            finish_reason=getattr(choice, "finish_reason", None),
            response_tokens=getattr(
                usage, "completion_tokens", None
            ),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
        )
