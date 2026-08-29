(run-config)=
# ⚙️ Run Configuration

`RunConfig` is the single place run behaviour lives. Everything about
*how* an evaluation executes is here, so metrics only have to know how
to score one row.

```python
from ragrank import evaluate
from ragrank.evaluation import RunConfig

evaluate(data, run_config=RunConfig(
    max_workers=8,
    max_retries=2,
    backoff=0.5,
    show_progress=True,
    raise_on_error=False,
    repetitions=1,
    reducer="mean",
    cache=True,
))
```

| Option | Default | What it does |
| --- | --- | --- |
| `max_workers` | 4 | Metric calls run concurrently. Set to 1 to run serially, which is easier to debug. |
| `max_retries` | 2 | Retries after a failing model call, on top of the retries a metric makes for an unreadable answer. |
| `backoff` | 0.5 | Seconds before the first retry, doubling each attempt. |
| `show_progress` | True | A progress bar, counting metric calls rather than rows. |
| `raise_on_error` | False | Abort on the first failure instead of recording it. |
| `repetitions` | 1 | Score each row this many times and reduce. |
| `reducer` | `"mean"` | How to reduce repetitions: `mean`, `median` or `mode`. |
| `cache` | None | Reuse identical prompts. See [caching](./caching.md). |

## A run survives its failures

By default a row that fails does not stop the evaluation. It comes back
with `score=None` and an `error` explaining why:

```python
result.failed_count           # how many (row, metric) pairs had no score
result.results[0][3].error    # and why that one did
```

This matters more than it sounds. A five thousand row evaluation that
dies forty minutes and twelve dollars in, because one response made the
judge say something unparseable, is a bad trade. Set
`raise_on_error=True` if you would rather it stopped.

## Validation happens first

Before a single call is made, Ragrank checks that every metric can
actually be satisfied by the data:

```text
ValidationError: The dataset cannot satisfy every metric:
'Correctness' needs ['reference'], which the dataset does not provide.
```

A missing column fails in milliseconds rather than part way through a
run you are paying for.

## Typos are errors

```python
RunConfig(max_worker=8)
# ValidationError: Unexpected keyword argument
```

An option that is silently ignored is worse than one that fails, since
it looks like it worked. Unknown options are rejected, and `reducer`
only accepts the values it actually supports.
