"""The pytest plugin, exercised through pytest's own test harness."""

from __future__ import annotations

import pytest

pytest_plugins = ["pytester"]

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
    result = pytester.runpytest("--strict-markers")
    result.assert_outcomes(passed=1)


def test_the_marker_appears_in_the_marker_list(
    pytester: pytest.Pytester,
) -> None:
    result = pytester.runpytest("--markers")
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
    pytester.runpytest("-m", "ragrank").assert_outcomes(passed=1)
    pytester.runpytest("-m", "not ragrank").assert_outcomes(passed=1)


def test_the_option_is_advertised(
    pytester: pytest.Pytester,
) -> None:
    result = pytester.runpytest("--help")
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
    pytester.runpytest().assert_outcomes(passed=1)


def test_the_eval_fixture_fails_a_bad_run(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(
        CONFTEST.replace('"0.9"', '"0.1"')
    )
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
    result = pytester.runpytest()
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
    result = pytester.runpytest()
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
    pytester.runpytest().assert_outcomes(passed=1)


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
    pytester.runpytest().assert_outcomes(passed=1)


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
    pytester.runpytest().assert_outcomes(passed=1)


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
    pytester.runpytest(
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
    pytester.runpytest().assert_outcomes(passed=1)
    assert not (pytester.path / "evals.html").exists()


def test_a_report_with_no_evals_still_writes(
    pytester: pytest.Pytester,
) -> None:
    pytester.makepyfile("def test_nothing():\n    assert True\n")
    report = pytester.path / "empty.html"
    pytester.runpytest(f"--ragrank-report={report}")

    assert "No evaluations ran" in report.read_text(
        encoding="utf-8"
    )
