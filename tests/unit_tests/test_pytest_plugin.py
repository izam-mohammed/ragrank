"""The pytest plugin, exercised through pytest's own test harness."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

# Every inner run goes through a subprocess. pytester's in-process
# runner snapshots and restores sys.modules, and numpy refuses to be
# initialised twice in one interpreter -- so the second inner run that
# imported ragrank would die with a bewildering pandas error about a
# missing dependency that is plainly installed.

# A conftest the generated suites share: a fake judge, so none of this
# needs a key or a network.
CONFTEST = """
import pytest
from ragrank.llm import FakeLLM

@pytest.fixture(scope="session")
def ragrank_llm():
    return FakeLLM(responses=["0.9"])
"""

# Indented to match the blocks it is concatenated with, so that
# pytester's dedent sees one consistent prefix.
DATA = """
        DATA = {
            "question": ["who wrote it"],
            "context": [["Ada wrote it."]],
            "response": ["Ada"],
        }
"""


def test_the_marker_is_registered(pytester: pytest.Pytester) -> None:
    """An unregistered marker is a warning, and --strict-markers an error."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.ragrank
        def test_marked():
            assert True
        """
    )
    result = pytester.runpytest_subprocess("--strict-markers")
    result.assert_outcomes(passed=1)


def test_the_marker_appears_in_the_marker_list(
    pytester: pytest.Pytester,
) -> None:
    result = pytester.runpytest_subprocess("--markers")
    result.stdout.fnmatch_lines(["*ragrank: an evaluation*"])


def test_evals_can_be_selected_and_skipped(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.ragrank
        def test_an_eval():
            assert True

        def test_a_unit_test():
            assert True
        """
    )
    pytester.runpytest_subprocess("-m", "ragrank").assert_outcomes(
        passed=1
    )
    pytester.runpytest_subprocess(
        "-m", "not ragrank"
    ).assert_outcomes(passed=1)


def test_the_option_is_advertised(
    pytester: pytest.Pytester,
) -> None:
    result = pytester.runpytest_subprocess("--help")
    result.stdout.fnmatch_lines(["*--ragrank-report*"])


def test_the_eval_fixture_passes_a_good_run(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        DATA
        + """
        import pytest
        from ragrank.metric import response_relevancy

        gate = response_relevancy.model_copy(
            update={"threshold": 0.5}
        )

        @pytest.mark.ragrank
        def test_it_is_relevant(ragrank_eval):
            result = ragrank_eval(DATA, [gate])
            assert result.passed is True
        """
    )
    pytester.runpytest_subprocess().assert_outcomes(passed=1)


def test_the_eval_fixture_fails_a_bad_run(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(CONFTEST.replace('"0.9"', '"0.1"'))
    pytester.makepyfile(
        DATA
        + """
        import pytest
        from ragrank.metric import response_relevancy

        gate = response_relevancy.model_copy(
            update={"threshold": 0.8}
        )

        @pytest.mark.ragrank
        def test_it_is_relevant(ragrank_eval):
            ragrank_eval(DATA, [gate])
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Evaluation failed*"])


def test_the_failure_points_at_the_test_not_at_ragrank(
    pytester: pytest.Pytester,
) -> None:
    """__tracebackhide__ keeps ragrank's own frames out of the report."""
    pytester.makeconftest(CONFTEST.replace('"0.9"', '"0.1"'))
    pytester.makepyfile(
        DATA
        + """
        import pytest
        from ragrank.metric import response_relevancy

        gate = response_relevancy.model_copy(
            update={"threshold": 0.8}
        )

        def test_it_is_relevant(ragrank_eval):
            ragrank_eval(DATA, [gate])
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(failed=1)
    assert "assertions.py" not in result.stdout.str()


def test_the_suite_judge_is_used(
    pytester: pytest.Pytester,
) -> None:
    """One conftest fixture points every eval at the same model."""
    pytester.makeconftest(CONFTEST.replace('"0.9"', '"0.42"'))
    pytester.makepyfile(
        DATA
        + """
        import pytest
        from ragrank.metric import response_relevancy

        gate = response_relevancy.model_copy(
            update={"threshold": 0.1}
        )

        def test_uses_the_suite_judge(ragrank_eval):
            result = ragrank_eval(DATA, [gate])
            assert result.scores == [[0.42]]
        """
    )
    pytester.runpytest_subprocess().assert_outcomes(passed=1)


def test_a_per_test_judge_wins_over_the_suite_one(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        DATA
        + """
        from ragrank.llm import FakeLLM
        from ragrank.metric import response_relevancy

        gate = response_relevancy.model_copy(
            update={"threshold": 0.1}
        )

        def test_overrides(ragrank_eval):
            result = ragrank_eval(
                DATA, [gate], llm=FakeLLM(responses=["0.33"])
            )
            assert result.scores == [[0.33]]
        """
    )
    pytester.runpytest_subprocess().assert_outcomes(passed=1)


def test_the_suite_run_config_is_used(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(
        CONFTEST
        + """
import pytest
from ragrank.evaluation import RunConfig

@pytest.fixture(scope="session")
def ragrank_run_config():
    return RunConfig(max_workers=1, show_progress=False)
"""
    )
    pytester.makepyfile(
        DATA
        + """
        from ragrank.metric import response_relevancy

        gate = response_relevancy.model_copy(
            update={"threshold": 0.1}
        )

        def test_runs(ragrank_eval):
            assert ragrank_eval(DATA, [gate]).passed is True
        """
    )
    pytester.runpytest_subprocess().assert_outcomes(passed=1)


def test_the_report_collects_every_run(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        DATA
        + """
        from ragrank.metric import response_relevancy

        gate = response_relevancy.model_copy(
            update={"threshold": 0.1}
        )

        def test_one(ragrank_eval):
            ragrank_eval(DATA, [gate])

        def test_two(ragrank_eval):
            ragrank_eval(DATA, [gate])
        """
    )
    report = pytester.path / "evals.html"
    pytester.runpytest_subprocess(
        f"--ragrank-report={report}"
    ).assert_outcomes(passed=2)

    html = report.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "test_one" in html
    assert "test_two" in html
    assert "Response Relevancy" in html


def test_no_report_is_written_without_the_option(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(
        DATA
        + """
        from ragrank.metric import response_relevancy

        gate = response_relevancy.model_copy(
            update={"threshold": 0.1}
        )

        def test_one(ragrank_eval):
            ragrank_eval(DATA, [gate])
        """
    )
    pytester.runpytest_subprocess().assert_outcomes(passed=1)
    assert not (pytester.path / "evals.html").exists()


def test_a_report_with_no_evals_still_writes(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile("def test_nothing():\n    assert True\n")
    report = pytester.path / "empty.html"
    pytester.runpytest_subprocess(f"--ragrank-report={report}")

    assert "No evaluations ran" in report.read_text(encoding="utf-8")


# The integration tests above run in subprocesses, which coverage
# cannot see. These exercise the same code in this process.

import ragrank_pytest  # noqa: E402


class FakeGroup:
    def __init__(self) -> None:
        self.options: list[tuple[tuple, dict]] = []

    def addoption(self, *args, **kwargs) -> None:
        self.options.append((args, kwargs))


class FakeParser:
    def __init__(self) -> None:
        self.group = FakeGroup()

    def getgroup(self, name: str) -> FakeGroup:
        assert name == "ragrank"
        return self.group


class FakeConfig:
    def __init__(self, report: str | None = None) -> None:
        self.lines: list[tuple[str, str]] = []
        self._report = report

    def addinivalue_line(self, name: str, line: str) -> None:
        self.lines.append((name, line))

    def getoption(self, name: str, default=None):
        assert name == "--ragrank-report"
        return self._report


def test_addoption_registers_the_report_flag() -> None:
    parser = FakeParser()
    ragrank_pytest.pytest_addoption(parser)

    ((args, kwargs),) = parser.group.options
    assert "--ragrank-report" in args
    assert kwargs["default"] is None


def test_configure_registers_the_marker_and_the_store() -> None:
    config = FakeConfig()
    ragrank_pytest.pytest_configure(config)

    assert any(
        name == "markers" and line.startswith("ragrank:")
        for name, line in config.lines
    )
    assert ragrank_pytest._recorded(config) == []


def test_the_store_is_created_on_demand() -> None:
    """A plugin hook can be skipped; the recorder must not assume."""
    config = FakeConfig()
    recorded = ragrank_pytest._recorded(config)

    assert recorded == []
    assert ragrank_pytest._recorded(config) is recorded


def test_the_default_judge_and_config_defer_to_the_library() -> None:
    assert ragrank_pytest.ragrank_llm.__wrapped__() is None
    assert ragrank_pytest.ragrank_run_config.__wrapped__() is None


def test_sessionfinish_does_nothing_without_the_option() -> None:
    class Session:
        config = FakeConfig(report=None)

    ragrank_pytest.pytest_sessionfinish(Session())


def test_sessionfinish_writes_the_report(tmp_path: Path) -> None:
    report = tmp_path / "out.html"

    class Session:
        config = FakeConfig(report=str(report))

    ragrank_pytest.pytest_sessionfinish(Session())

    assert "No evaluations ran" in report.read_text(encoding="utf-8")


def test_the_fixture_records_the_run_it_asserted() -> None:
    from ragrank.dataset import from_dict
    from ragrank.llm import FakeLLM
    from ragrank.metric import response_relevancy

    config = FakeConfig()
    ragrank_pytest.pytest_configure(config)

    class Node:
        nodeid = "tests/test_x.py::test_y"

    class Request:
        pass

    request = Request()
    request.config = config
    request.node = Node()

    run = next(
        ragrank_pytest.ragrank_eval.__wrapped__(
            request, FakeLLM(responses=["0.9"]), None
        )
    )
    data = from_dict(
        {
            "question": ["q"],
            "context": [["c"]],
            "response": ["r"],
        },
        return_as_dataset=True,
    )
    gate = response_relevancy.model_copy(update={"threshold": 0.5})

    result = run(data, [gate])

    assert result.passed is True
    assert ragrank_pytest._recorded(config) == [
        ("tests/test_x.py::test_y", result)
    ]


def test_a_failing_run_is_not_recorded() -> None:
    """The assertion fires first, so nothing lands in the report."""
    from ragrank.dataset import from_dict
    from ragrank.llm import FakeLLM
    from ragrank.metric import response_relevancy
    from ragrank.testing import MetricAssertionError

    config = FakeConfig()
    ragrank_pytest.pytest_configure(config)

    class Node:
        nodeid = "tests/test_x.py::test_y"

    class Request:
        pass

    request = Request()
    request.config = config
    request.node = Node()

    run = next(
        ragrank_pytest.ragrank_eval.__wrapped__(
            request, FakeLLM(responses=["0.1"]), None
        )
    )
    data = from_dict(
        {
            "question": ["q"],
            "context": [["c"]],
            "response": ["r"],
        },
        return_as_dataset=True,
    )
    gate = response_relevancy.model_copy(update={"threshold": 0.9})

    with pytest.raises(MetricAssertionError):
        run(data, [gate])

    assert ragrank_pytest._recorded(config) == []
