(the-testing)=
# ✅ Testing

Evaluations belong in your test suite, next to your unit tests, rather
than in a script somebody remembers to run.

```python
from ragrank.dataset import DataNode
from ragrank.metric import faithfulness
from ragrank.testing import assert_metric


def test_the_bot_stays_grounded():
    node = DataNode(
        question="Where is the Eiffel Tower?",
        context=["The Eiffel Tower is in Paris."],
        response="It is in Paris.",
    )
    assert_metric(node, faithfulness, threshold=0.9)
```

Run it with `pytest`, or `unittest`, or anything else that understands
an assertion. There is no custom runner and no `ragrank test run`
command, so every pytest flag and plugin you already use keeps working.

Installing ragrank also registers a small [pytest plugin](#the-pytest-plugin),
which adds a marker and a report. It is optional - the assertions above
work with or without it.

## The three helpers

### One datapoint

```python
from ragrank.testing import assert_metric

assert_metric(node, response_relevancy, threshold=0.7)
```

(a-whole-dataset)=
### A whole dataset

```python
from ragrank.testing import assert_evaluation

result = assert_evaluation(dataset, [strict_relevancy])
```

Returns the `EvalResult` so the test can inspect it further. At least
one metric must carry a threshold, otherwise there is nothing to
assert and it says so.

### No regression against a baseline

```python
from ragrank.testing import assert_no_regression

assert_no_regression(last_release, this_branch)
```

Changes within the noise pass. The point is to catch real regressions,
not to demand that a number never moves. See
[comparing runs](../evaluation/comparing_runs.md).

## Failures explain themselves

The assertion carries the diagnosis the metric already computed:

```text
MetricAssertionError: Faithfulness scored 0.500, below the threshold of 0.900.
  Unsupported claims:
    - It was built in 1750.
```

A chunkwise metric shows its per chunk scores instead. That is the
difference between a test that tells you something failed and a test
that tells you what to fix.

## Two deliberate choices

**A metric that could not score fails rather than passing.** Silence is
not success, so an unscorable row is a test failure with the reason
attached.

**Asserting with no threshold is an error.** A test that cannot fail is
worse than no test, so `assert_metric` raises `ValueError` rather than
passing vacuously.

(the-pytest-plugin)=
## The pytest plugin

Installing ragrank registers it automatically. There is nothing to add
to a conftest.

### Marking evals

Evaluations are slow and cost money; unit tests are neither. The
`ragrank` marker lets you keep them apart:

```python
import pytest


@pytest.mark.ragrank
def test_the_bot_stays_grounded(ragrank_eval):
    ragrank_eval(dataset, [faithfulness_gate])
```

```bash
pytest -m "not ragrank"     # the fast suite, on every commit
pytest -m ragrank           # the evals, nightly
```

### The `ragrank_eval` fixture

Wraps [`assert_evaluation`](#a-whole-dataset) with the suite's judge and
run config already applied, and records the result for the report.

```python
def test_it_answers_well(ragrank_eval):
    result = ragrank_eval(dataset, [strict_relevancy])
    assert result.usage.total_tokens < 10_000
```

### Setting the judge once

Two session fixtures, overridden in a `conftest.py`, configure every
eval in the suite:

```python
import pytest

from ragrank.evaluation import RunConfig
from ragrank.integrations.litellm import LiteLLM


@pytest.fixture(scope="session")
def ragrank_llm():
    return LiteLLM(model="anthropic/claude-sonnet-4-5")


@pytest.fixture(scope="session")
def ragrank_run_config():
    return RunConfig(max_workers=8, cache=True)
```

A single test can still override either by passing `llm=` or
`run_config=` to `ragrank_eval`.

### A report for the whole session

```bash
pytest -m ragrank --ragrank-report=evals.html
```

Every evaluation becomes a section of one
[HTML report](../evaluation/reports.md), named by its test. Useful as a
CI artefact when a nightly eval run fails and somebody has to work out
which rows caused it.

### Tracebacks point at your test

The assertion helpers hide their own frames, so a failed eval reports
the line you wrote rather than the inside of `assertions.py`.

## Testing without spending anything

`FakeLLM` returns scripted responses, so the whole pipeline can be
exercised in CI with no API key and no cost:

```python
from ragrank.llm import FakeLLM

assert_metric(
    node,
    response_relevancy,
    threshold=0.7,
    llm=FakeLLM(responses=["0.9"]),
)
```

Combine that with the [free metrics](../metrics/heuristic_metrics/index.md)
and a full evaluation runs in CI for nothing.
