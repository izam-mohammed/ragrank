"""Assertions for putting evaluations in your test suite.

Evals belong next to unit tests, not in a separate script somebody runs
by hand. These are plain assertions -- no runner, no plugin, no custom
CLI -- so they work with pytest, unittest, or anything that understands
`assert`.

    from ragrank.testing import assert_metric

    def test_bot_answers_relevantly():
        assert_metric(
            DataNode(question=..., context=[...], response=...),
            response_relevancy,
            threshold=0.7,
        )
"""

from ragrank.testing.assertions import (
    MetricAssertionError,
    assert_evaluation,
    assert_metric,
    assert_no_regression,
)

__all__ = [
    "assert_metric",
    "assert_evaluation",
    "assert_no_regression",
    "MetricAssertionError",
]
