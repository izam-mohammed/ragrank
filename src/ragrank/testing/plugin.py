"""A pytest plugin, so evals are just tests.

`ragrank.testing` already had the assertions. What it lacked was any
way for pytest to know they existed -- which meant no way to run only
the evals, no way to skip them when they cost money, and a failure
traceback that pointed into ragrank's internals rather than at the line
the developer wrote.

Installed automatically through the `pytest11` entry point; there is
nothing to add to a conftest.

Marking evals::

    @pytest.mark.ragrank
    def test_the_bot_stays_grounded(ragrank_eval):
        ragrank_eval(dataset, [faithfulness_gate])

Then `pytest -m "not ragrank"` keeps the fast suite fast, and
`pytest -m ragrank --ragrank-report=evals.html` produces something to
attach to the pull request.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import pytest

from ragrank.dataset import DataNode, Dataset
from ragrank.evaluation.outputs import EvalResult
from ragrank.evaluation.runner import RunConfig
from ragrank.llm import BaseLLM
from ragrank.metric import BaseMetric
from ragrank.testing.assertions import assert_evaluation

if TYPE_CHECKING:  # pragma: no cover
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser

#: Where recorded runs accumulate for the session-end report.
STASH_KEY = "_ragrank_recorded"


def pytest_addoption(parser: Parser) -> None:
    """Register ragrank's command line options.

    Args:
        parser (Parser): The pytest argument parser.
    """
    group = parser.getgroup("ragrank")
    group.addoption(
        "--ragrank-report",
        action="store",
        default=None,
        metavar="PATH",
        help=(
            "write an HTML report of every ragrank evaluation run "
            "during this session"
        ),
    )


def pytest_configure(config: Config) -> None:
    """Register the marker and prepare the recorder.

    Args:
        config (Config): The pytest config object.
    """
    config.addinivalue_line(
        "markers",
        "ragrank: an evaluation, which may be slow and cost money. "
        "Select with -m ragrank, or skip with -m 'not ragrank'.",
    )
    setattr(config, STASH_KEY, [])


def _recorded(config: Config) -> list[tuple[str, EvalResult]]:
    """The runs recorded so far this session.

    Args:
        config (Config): The pytest config object.

    Returns:
        list[tuple[str, EvalResult]]: Named runs, in order.
    """
    recorded = getattr(config, STASH_KEY, None)
    if recorded is None:
        recorded = []
        setattr(config, STASH_KEY, recorded)
    return recorded


@pytest.fixture(scope="session")
def ragrank_llm() -> BaseLLM | None:
    """The model evals judge with.

    Defaults to None, which lets each metric fall back to the library
    default. Override it in a conftest to point every eval in the suite
    at one judge, or at a fake:

        @pytest.fixture(scope="session")
        def ragrank_llm():
            return LiteLLM(model="anthropic/claude-sonnet-4-5")

    Returns:
        BaseLLM | None: The judge, or None for the default.
    """
    return None


@pytest.fixture(scope="session")
def ragrank_run_config() -> RunConfig | None:
    """How evals in this suite execute.

    Override in a conftest to set concurrency, caching or repetitions
    once for every eval rather than per test.

    Returns:
        RunConfig | None: The run policy, or None for the default.
    """
    return None


@pytest.fixture
def ragrank_eval(
    request: pytest.FixtureRequest,
    ragrank_llm: BaseLLM | None,
    ragrank_run_config: RunConfig | None,
) -> Iterator[Callable[..., EvalResult]]:
    """Run an evaluation, assert its thresholds, and record it.

    Wraps `assert_evaluation` with the suite's judge and run config
    already applied, and files the result for `--ragrank-report`.

    Args:
        request (pytest.FixtureRequest): The active test request.
        ragrank_llm (BaseLLM | None): The suite's judge.
        ragrank_run_config (RunConfig | None): The suite's run policy.

    Yields:
        Callable[..., EvalResult]: The evaluating assertion.
    """

    def run(
        data: Dataset | DataNode | dict,
        metrics: BaseMetric | list[BaseMetric],
        *,
        llm: BaseLLM | None = None,
        run_config: RunConfig | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> EvalResult:
        __tracebackhide__ = True
        result = assert_evaluation(
            data,
            metrics,
            llm=llm or ragrank_llm,
            run_config=run_config or ragrank_run_config,
            **kwargs,
        )
        _recorded(request.config).append(
            (request.node.nodeid, result)
        )
        return result

    yield run


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session) -> None:
    """Write the HTML report, if one was asked for.

    Args:
        session (pytest.Session): The finished session.
    """
    path = session.config.getoption("--ragrank-report", None)
    if not path:
        return

    from pathlib import Path

    from ragrank.evaluation.report import combined_html

    Path(path).write_text(
        combined_html(
            _recorded(session.config), title="ragrank evaluations"
        ),
        encoding="utf-8",
    )
