"""Tests for integration logic that does not need a live provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from ragrank.dataset import ColumnMap, Dataset, from_hfdataset

# --------------------------- hugging face ---------------------------


def test_from_hfdataset_builds_a_dataset() -> None:
    """The loader maps HF columns onto a ragrank Dataset."""
    rows = {
        "question": ["q1", "q2"],
        "context": ["['c1']", "['c2', 'c3']"],
        "response": ["r1", "r2"],
    }
    split = MagicMock()
    split.column_names = list(rows)
    split.__getitem__.side_effect = rows.__getitem__

    fake_datasets = SimpleNamespace(
        load_dataset=MagicMock(return_value={"train": split})
    )

    with patch(
        "ragrank.utils.optional.import_module",
        return_value=fake_datasets,
    ):
        dataset = from_hfdataset("some/dataset", split="train")

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert dataset[0].context == ["c1"]
    assert dataset[1].context == ["c2", "c3"]
    fake_datasets.load_dataset.assert_called_once_with(
        "some/dataset"
    )


def test_from_hfdataset_accepts_a_tuple_url() -> None:
    """A (repo, config) tuple is unpacked into load_dataset."""
    rows = {
        "question": ["q"],
        "context": ["['c']"],
        "response": ["r"],
    }
    split = MagicMock()
    split.column_names = list(rows)
    split.__getitem__.side_effect = rows.__getitem__

    fake_datasets = SimpleNamespace(
        load_dataset=MagicMock(return_value={"test": split})
    )

    with patch(
        "ragrank.utils.optional.import_module",
        return_value=fake_datasets,
    ):
        from_hfdataset(("repo", "config"), split="test")

    fake_datasets.load_dataset.assert_called_once_with(
        "repo", "config"
    )


def test_from_hfdataset_honours_a_column_map() -> None:
    """Renamed HF columns are mapped across."""
    rows = {
        "query": ["q"],
        "context": ["['c']"],
        "answer": ["r"],
    }
    split = MagicMock()
    split.column_names = list(rows)
    split.__getitem__.side_effect = rows.__getitem__

    fake_datasets = SimpleNamespace(
        load_dataset=MagicMock(return_value={"train": split})
    )

    with patch(
        "ragrank.utils.optional.import_module",
        return_value=fake_datasets,
    ):
        dataset = from_hfdataset(
            "repo",
            split="train",
            column_map=ColumnMap(
                question="query", response="answer"
            ),
        )

    assert dataset[0].question == "q"
    assert dataset[0].response == "r"


# --------------------------- langchain ---------------------------

langchain_core = pytest.importorskip("langchain_core")


@pytest.mark.parametrize(
    ("llm_output", "expected"),
    [
        ({"token_usage": {"completion_tokens": 42}}, 42),
        ({"token_usage": {}}, None),
        ({}, None),
        (None, None),
        ({"token_usage": None}, None),
        ({"token_usage": {"completion_tokens": "12"}}, None),
    ],
)
def test_completion_tokens_is_defensive(
    llm_output: dict | None, expected: int | None
) -> None:
    """Issue #46: most non-OpenAI LangChain models report no usage.

    Reading llm_output["token_usage"]["completion_tokens"] outright
    raised KeyError for every one of them.
    """
    from ragrank.integrations.langchain.langchain_llm_wrapper import (
        _completion_tokens,
    )

    result = SimpleNamespace(llm_output=llm_output)
    assert _completion_tokens(result) == expected


def test_ragrank_prompt_value_round_trip() -> None:
    """The LangChain prompt wrapper carries the text through."""
    from ragrank.integrations.langchain.langchain_llm_wrapper import (
        RagrankPromptValue,
    )

    prompt = RagrankPromptValue(prompt_str="hello there")
    assert prompt.to_string() == "hello there"
    assert prompt.to_messages()[0].content == "hello there"


def test_langchain_wrapper_rejects_a_non_langchain_llm() -> None:
    """A clear TypeError beats a confusing failure later."""
    from ragrank.integrations.langchain import LangchainLLMWrapper

    with pytest.raises(Exception, match="not valid"):
        LangchainLLMWrapper(llm="not an llm")


# --------------------------- openai ---------------------------


def test_openai_wrapper_names_its_model() -> None:
    """The model is configurable and shows up in the name."""
    pytest.importorskip("openai")
    from ragrank.integrations.openai import OpenaiLLM

    assert "gpt-4o-mini" in OpenaiLLM().name
    assert "gpt-4.1" in OpenaiLLM(model="gpt-4.1").name


def test_openai_wrapper_needs_a_key_only_when_used() -> None:
    """Constructing must not require credentials; calling does."""
    pytest.importorskip("openai")
    from ragrank.integrations.openai import OpenaiLLM

    llm = OpenaiLLM()
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(ValueError, match="is not set"),
    ):
        _ = llm.client


def test_openai_client_is_built_once() -> None:
    """A client per call re-read the environment for every row."""
    pytest.importorskip("openai")
    from ragrank.integrations.openai import OpenaiLLM

    llm = OpenaiLLM()
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        assert llm.client is llm.client


def test_openai_generate_text_matches_the_base_signature() -> None:
    """Issue #46: extra kwargs broke the BaseLLM contract."""
    pytest.importorskip("openai")
    from inspect import signature

    from ragrank.integrations.openai import OpenaiLLM
    from ragrank.llm import BaseLLM

    assert list(
        signature(OpenaiLLM.generate_text).parameters
    ) == list(signature(BaseLLM.generate_text).parameters)
