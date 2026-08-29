"""A deterministic in-process LLM, for tests and offline experimentation.

`FakeLLM` never touches the network and needs no credentials, so the whole
library can be exercised end to end without an API key.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import cycle
from threading import Lock

from ragrank.bridge.pydantic import ConfigDict, Field, PrivateAttr
from ragrank.llm.base import BaseLLM, LLMResult


class FakeLLM(BaseLLM):
    """A scripted, deterministic LLM.

    Attributes:
        responses (list[str]): Responses handed out in order. Once exhausted
            the list repeats, so a single-element script answers every call.
            Which row receives which response is only deterministic when
            the run is serial; use `response_fn` to key answers on the
            prompt instead.
        response_fn (Callable[[str], str] | None): Takes precedence over
            `responses` when set; receives the prompt and returns the
            response.

    Examples::

        from ragrank.llm.fake import FakeLLM

        llm = FakeLLM(responses=["0.8", "0.2"])
        llm.generate_text("anything").response
    """

    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True
    )

    responses: list[str] = Field(
        default_factory=lambda: ["0.5"],
        description="Responses handed out in order, cycling when exhausted.",
    )
    response_fn: Callable[[str], str] | None = Field(
        default=None,
        repr=False,
        description="Called with the prompt to produce the response.",
    )

    _cursor: Iterator[str] | None = PrivateAttr(default=None)
    _prompts: list[str] = PrivateAttr(default_factory=list)
    _lock: Lock = PrivateAttr(default_factory=Lock)

    @property
    def name(self) -> str:
        """Get the name of the language model.

        Returns:
            str: The name of the language model.
        """
        return "Fake LLM"

    @property
    def prompts(self) -> list[str]:
        """Every prompt this LLM has been asked to complete.

        Returns:
            list[str]: The prompts, in call order.
        """
        return list(self._prompts)

    def generate_text(self, text: str) -> LLMResult:
        """Return the next scripted response.

        Args:
            text (str): The input text.

        Returns:
            LLMResult: The scripted result.
        """
        if self.response_fn is not None:
            with self._lock:
                self._prompts.append(text)
            message = self.response_fn(text)
        else:
            with self._lock:
                self._prompts.append(text)
                if self._cursor is None:
                    if not self.responses:
                        raise ValueError(
                            "FakeLLM needs at least one response."
                        )
                    self._cursor = cycle(self.responses)
                message = next(self._cursor)

        return LLMResult(
            response=message,
            response_time=0.0,
            finish_reason="stop",
            response_tokens=len(message.split()),
            prompt_tokens=len(text.split()),
            llm=self,
            llm_config=self.llm_config,
        )
