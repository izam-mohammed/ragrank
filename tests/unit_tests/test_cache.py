"""Tests for judge response caching."""

from __future__ import annotations

from pathlib import Path

import pytest
from ragrank import evaluate
from ragrank.dataset import Dataset
from ragrank.evaluation import RunConfig
from ragrank.llm import (
    CachedLLM,
    DiskCache,
    FakeLLM,
    LLMConfig,
    MemoryCache,
)
from ragrank.llm.cache import CacheBackend, cache_key
from ragrank.metric import (
    exact_match,
    response_conciseness,
    response_relevancy,
)


@pytest.fixture
def dataset() -> Dataset:
    """Five rows."""
    return Dataset(
        question=[f"q{i}" for i in range(5)],
        context=[["c"]] * 5,
        response=["r"] * 5,
        reference=["r"] * 5,
    )


def config(cache: object) -> RunConfig:
    """A serial run with the given cache."""
    return RunConfig(show_progress=False, max_workers=1, cache=cache)


# --------------------------- the key ---------------------------


def test_same_request_same_key() -> None:
    """Identical requests must collide."""
    args = ("gpt-4o-mini", LLMConfig(), "a prompt")
    assert cache_key(*args) == cache_key(*args)


def test_prompt_changes_the_key() -> None:
    """A different question is a different request."""
    assert cache_key("m", LLMConfig(), "one") != cache_key(
        "m", LLMConfig(), "two"
    )


def test_model_changes_the_key() -> None:
    """The same prompt to a different model is a different request."""
    assert cache_key("a", LLMConfig(), "p") != cache_key(
        "b", LLMConfig(), "p"
    )


def test_settings_that_change_the_answer_change_the_key() -> None:
    """Temperature is part of the request, so part of the key."""
    assert cache_key(
        "m", LLMConfig(temperature=0.0), "p"
    ) != cache_key("m", LLMConfig(temperature=1.0), "p")


# --------------------------- backends ---------------------------


def test_memory_cache_round_trip() -> None:
    """Store then fetch."""
    cache = MemoryCache()
    assert cache.get("missing") is None
    cache.set("k", "v")
    assert cache.get("k") == "v"
    assert len(cache) == 1


def test_disk_cache_round_trip(tmp_path: Path) -> None:
    """Store then fetch, from disk."""
    cache = DiskCache(tmp_path / "cache")
    assert cache.get("missing") is None
    cache.set("k", "v")
    assert cache.get("k") == "v"
    assert len(cache) == 1


def test_disk_cache_survives_a_new_instance(tmp_path: Path) -> None:
    """The point of a disk cache: it outlives the process."""
    DiskCache(tmp_path).set("k", "v")
    assert DiskCache(tmp_path).get("k") == "v"


def test_corrupt_disk_entry_is_a_miss(tmp_path: Path) -> None:
    """A broken cache must never break a run."""
    cache = DiskCache(tmp_path)
    (tmp_path / "deadbeef.json").write_text("not json")
    assert cache.get("deadbeef") is None


def test_unwritable_disk_cache_does_not_raise(
    tmp_path: Path,
) -> None:
    """Losing a cache entry is not worth losing a run over."""
    cache = DiskCache(tmp_path)
    cache.directory = tmp_path / "does" / "not" / "exist"
    cache.set("k", "v")  # must not raise
    assert cache.get("k") is None


def test_cache_backend_base_is_abstract() -> None:
    """Subclasses must implement both halves."""
    with pytest.raises(NotImplementedError):
        CacheBackend().get("k")
    with pytest.raises(NotImplementedError):
        CacheBackend().set("k", "v")


# --------------------------- CachedLLM ---------------------------


def test_cached_llm_serves_repeats_without_calling() -> None:
    """The second identical prompt must not reach the model."""
    inner = FakeLLM(responses=["0.5"])
    llm = CachedLLM(inner=inner, backend=MemoryCache())

    first = llm.generate_text("same prompt")
    second = llm.generate_text("same prompt")

    assert first.response == second.response == "0.5"
    assert len(inner.prompts) == 1
    assert llm.hits == 1
    assert llm.misses == 1


def test_cached_result_reports_zero_cost() -> None:
    """A cached answer costs nothing, and says so."""
    llm = CachedLLM(
        inner=FakeLLM(responses=["0.5"]), backend=MemoryCache()
    )
    llm.generate_text("p")
    cached = llm.generate_text("p")

    assert cached.prompt_tokens == 0
    assert cached.response_tokens == 0
    assert cached.response_time == 0.0
    assert cached.finish_reason == "cached"


def test_different_prompts_are_not_confused() -> None:
    """Cache hits must be exact."""
    inner = FakeLLM(responses=["a", "b"])
    llm = CachedLLM(inner=inner, backend=MemoryCache())
    assert llm.generate_text("one").response == "a"
    assert llm.generate_text("two").response == "b"
    assert llm.misses == 2


def test_cached_llm_keeps_the_inner_name() -> None:
    """Wrapping must be transparent to reporting."""
    assert (
        CachedLLM(inner=FakeLLM(), backend=MemoryCache()).name
        == "Fake LLM"
    )


# --------------------------- in a run ---------------------------


def test_second_run_costs_nothing(dataset: Dataset) -> None:
    """The headline benefit."""
    cache = MemoryCache()
    llm = FakeLLM(responses=["0.5"])

    first = evaluate(
        dataset,
        llm=llm,
        metrics=[response_relevancy],
        run_config=config(cache),
    )
    second = evaluate(
        dataset,
        llm=llm,
        metrics=[response_relevancy],
        run_config=config(cache),
    )

    assert first.usage.total_tokens > 0
    assert second.usage.total_tokens == 0
    assert first.scores == second.scores


def test_adding_a_metric_only_pays_for_that_metric(
    dataset: Dataset,
) -> None:
    """The everyday case: change one thing, re-pay for one thing."""
    cache = MemoryCache()
    llm = FakeLLM(responses=["0.5"])

    evaluate(
        dataset,
        llm=llm,
        metrics=[response_relevancy],
        run_config=config(cache),
    )
    both = evaluate(
        dataset,
        llm=llm,
        metrics=[response_relevancy, response_conciseness],
        run_config=config(cache),
    )

    only_new = evaluate(
        dataset,
        llm=FakeLLM(responses=["0.5"]),
        metrics=[response_conciseness],
        run_config=config(MemoryCache()),
    )
    assert both.usage.total_tokens == only_new.usage.total_tokens


def test_caching_is_off_by_default(dataset: Dataset) -> None:
    """No surprise files on disk unless asked for."""
    assert RunConfig().cache is None
    llm = FakeLLM(responses=["0.5"])
    first = evaluate(
        dataset,
        llm=llm,
        metrics=[response_relevancy],
        run_config=RunConfig(show_progress=False, max_workers=1),
    )
    second = evaluate(
        dataset,
        llm=llm,
        metrics=[response_relevancy],
        run_config=RunConfig(show_progress=False, max_workers=1),
    )
    assert second.usage.total_tokens == first.usage.total_tokens


def test_cache_true_uses_disk(
    dataset: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`cache=True` is the one-flag path."""
    monkeypatch.chdir(tmp_path)
    llm = FakeLLM(responses=["0.5"])
    evaluate(
        dataset,
        llm=llm,
        metrics=[response_relevancy],
        run_config=config(True),
    )
    second = evaluate(
        dataset,
        llm=llm,
        metrics=[response_relevancy],
        run_config=config(True),
    )
    assert (tmp_path / ".ragrank_cache").is_dir()
    assert second.usage.total_tokens == 0


def test_cache_false_disables(dataset: Dataset) -> None:
    """False is as good as None."""
    llm = FakeLLM(responses=["0.5"])
    evaluate(
        dataset,
        llm=llm,
        metrics=[response_relevancy],
        run_config=config(False),
    )
    second = evaluate(
        dataset,
        llm=llm,
        metrics=[response_relevancy],
        run_config=config(False),
    )
    assert second.usage.total_tokens > 0


def test_deterministic_metrics_ignore_the_cache(
    dataset: Dataset,
) -> None:
    """Nothing to cache when nothing is called."""
    cache = MemoryCache()
    evaluate(
        dataset,
        llm=FakeLLM(),
        metrics=[exact_match],
        run_config=config(cache),
    )
    assert len(cache) == 0


def test_cache_is_safe_under_concurrency(dataset: Dataset) -> None:
    """Shared across threads, so it must not corrupt or lose entries."""
    cache = MemoryCache()
    big = Dataset(
        question=[f"q{i}" for i in range(40)],
        context=[["c"]] * 40,
        response=["r"] * 40,
    )
    first = evaluate(
        big,
        llm=FakeLLM(responses=["0.5"]),
        metrics=[response_relevancy],
        run_config=RunConfig(
            show_progress=False, max_workers=8, cache=cache
        ),
    )
    second = evaluate(
        big,
        llm=FakeLLM(responses=["0.5"]),
        metrics=[response_relevancy],
        run_config=RunConfig(
            show_progress=False, max_workers=8, cache=cache
        ),
    )
    assert len(cache) == 40
    assert second.usage.total_tokens == 0
    assert first.scores == second.scores
