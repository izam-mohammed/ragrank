"""The LiteLLM judge, without a provider or a key in sight."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from ragrank.integrations.litellm import LiteLLM
from ragrank.llm import LLMConfig


def fake_completion(
    content: str = "0.9",
    *,
    prompt_tokens: int = 11,
    completion_tokens: int = 3,
) -> SimpleNamespace:
    """Build a LiteLLM-shaped completion object."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def patched_litellm(completion: object) -> MagicMock:
    """A stand-in litellm module returning `completion`."""
    module = MagicMock()
    module.completion.return_value = completion
    return module


def test_generate_text_reads_the_completion() -> None:
    module = patched_litellm(fake_completion("0.75"))
    llm = LiteLLM(model="anthropic/claude-sonnet-4-5")

    with patch(
        "ragrank.utils.optional.import_module", return_value=module
    ):
        result = llm.generate_text("rate this")

    assert result.response == "0.75"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 11
    assert result.response_tokens == 3
    assert result.llm is llm


def test_the_model_string_reaches_litellm() -> None:
    module = patched_litellm(fake_completion())
    llm = LiteLLM(model="gemini/gemini-2.0-flash")

    with patch(
        "ragrank.utils.optional.import_module", return_value=module
    ):
        llm.generate_text("hello")

    kwargs = module.completion.call_args.kwargs
    assert kwargs["model"] == "gemini/gemini-2.0-flash"
    assert kwargs["messages"][1]["content"] == "hello"


def test_the_judge_config_is_forwarded() -> None:
    module = patched_litellm(fake_completion())
    llm = LiteLLM()
    llm.set_config(LLMConfig(temperature=0.0, max_tokens=64))

    with patch(
        "ragrank.utils.optional.import_module", return_value=module
    ):
        llm.generate_text("hello")

    kwargs = module.completion.call_args.kwargs
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 64


def test_unset_options_are_not_sent() -> None:
    """Providers differ; sending an explicit None is rejected by some."""
    module = patched_litellm(fake_completion())

    with patch(
        "ragrank.utils.optional.import_module", return_value=module
    ):
        LiteLLM().generate_text("hello")

    kwargs = module.completion.call_args.kwargs
    assert "stop" not in kwargs
    assert "api_key" not in kwargs
    assert "api_base" not in kwargs


def test_credentials_and_endpoint_are_sent_when_given() -> None:
    module = patched_litellm(fake_completion())
    llm = LiteLLM(
        model="ollama/llama3",
        api_key="sk-test",
        api_base="http://localhost:11434",
    )

    with patch(
        "ragrank.utils.optional.import_module", return_value=module
    ):
        llm.generate_text("hello")

    kwargs = module.completion.call_args.kwargs
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["api_base"] == "http://localhost:11434"


def test_extra_params_are_forwarded() -> None:
    module = patched_litellm(fake_completion())
    llm = LiteLLM(extra_params={"api_version": "2024-02-01"})

    with patch(
        "ragrank.utils.optional.import_module", return_value=module
    ):
        llm.generate_text("hello")

    assert (
        module.completion.call_args.kwargs["api_version"]
        == "2024-02-01"
    )


def test_a_stop_sequence_is_forwarded_when_set() -> None:
    module = patched_litellm(fake_completion())
    llm = LiteLLM()
    llm.set_config(LLMConfig(stop=["\n"]))

    with patch(
        "ragrank.utils.optional.import_module", return_value=module
    ):
        llm.generate_text("hello")

    assert module.completion.call_args.kwargs["stop"] == ["\n"]


def test_no_choices_is_an_error_naming_the_model() -> None:
    module = patched_litellm(SimpleNamespace(choices=[], usage=None))

    with (
        patch(
            "ragrank.utils.optional.import_module",
            return_value=module,
        ),
        pytest.raises(ValueError, match="anthropic/claude"),
    ):
        LiteLLM(model="anthropic/claude-sonnet-4-5").generate_text(
            "hi"
        )


def test_missing_usage_is_not_fatal() -> None:
    module = patched_litellm(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="0.5"),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
    )

    with patch(
        "ragrank.utils.optional.import_module", return_value=module
    ):
        result = LiteLLM().generate_text("hi")

    assert result.response == "0.5"
    assert result.prompt_tokens is None


def test_empty_content_becomes_an_empty_string() -> None:
    module = patched_litellm(fake_completion(content=None))

    with patch(
        "ragrank.utils.optional.import_module", return_value=module
    ):
        assert LiteLLM().generate_text("hi").response == ""


def test_a_missing_litellm_explains_the_extra() -> None:
    with (
        patch(
            "ragrank.utils.optional.import_module",
            side_effect=ModuleNotFoundError,
        ),
        pytest.raises(
            ModuleNotFoundError, match=r"ragrank\[litellm\]"
        ),
    ):
        LiteLLM().generate_text("hi")


def test_the_name_carries_the_model() -> None:
    assert "gemini" in LiteLLM(model="gemini/pro").name
