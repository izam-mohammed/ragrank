(comparing-runs)=
# 📊 Comparing Runs

Nobody's workflow is a single evaluation. The question is always "is
this better than what I had?".

```python
from ragrank.evaluation import compare

baseline = evaluate(data, metrics=metrics)
candidate = evaluate(data, metrics=metrics, llm=new_model)

print(compare(baseline, candidate))
```

```text
Response Relevancy: 0.517 -> 0.900 (+0.383) [significant]
Faithfulness:       0.812 -> 0.815 (+0.003) [within noise]
```

## It tells you when a change is noise

This is the part that matters. Ragrank already reports a standard error
for every metric, so `compare` can say whether a difference is larger
than the spread of the runs it came from.

A change smaller than roughly two combined standard errors is labelled
`[within noise]`. 0.003 is not an improvement, and a library that lets
you claim it is one is not helping you.

The test is deliberately conservative and deliberately rough. It is a
guard against over-reading a small number, not a formal statistical
claim.

## It tells you which rows moved

```python
delta = compare(baseline, candidate).deltas[0]

delta.improved_rows    # [3, 7, 12]
delta.regressed_rows   # [5]
```

An aggregate that dropped by 0.4 is something you can go and look at.
Rows that were unscored in either run are skipped rather than counted
as movement.

## Using it as a gate

```python
diff = compare(baseline, candidate)
assert not diff.regressed
```

`regressed` lists only the metrics that got **significantly** worse, so
a build does not fail because a number wobbled. `improved` is its
counterpart, and `bool(diff)` is `True` only when something moved
beyond the noise.

For a ready-made assertion see
[the testing helpers](../testing/index.md).

## Only shared metrics are compared

Adding a metric to a run is not a regression in the metrics you already
had, so metrics missing from either side are skipped.

## From the command line

```bash
ragrank eval config.json --output before.json
# change something
ragrank eval config.json --output after.json
ragrank compare before.json after.json
```
