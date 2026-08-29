(judge-alignment)=
# 🔍 Is the judge any good?

Every number this library produces rests on one assumption: that the
judge agrees with a person. Almost nothing in the space measures that,
which is why *why should I trust these scores* is the first question
anyone sensible asks.

Label a few dozen rows by hand, run the judge over the same rows, and
compare.

```python
from ragrank.evaluation import align

result = evaluate(dataset, metrics=[faithfulness])
report = align(result, human_labels, threshold=0.5)

print(report)
```

```text
Faithfulness vs human labels (n=40)
  pearson: +0.812
  spearman: +0.799
  mean abs error: +0.140
  bias: +0.115
  agreement: +0.850
  kappa: +0.694
```

## Reading it

| Statistic | What it tells you |
| --- | --- |
| `pearson` | Do the scores move together, on the same scale? |
| `spearman` | Do they rank the rows the same way? |
| `mean_absolute_error` | How big is a typical disagreement? |
| `bias` | Which way does the judge lean? |
| `agreement` | How often do they land on the same side of the threshold? |
| `kappa` | The same, discounting the agreement you would get by chance |

**`bias` is the one to look at first.** A judge that is generous by a
consistent 0.115 is a prompt you can fix, or a threshold you can shift.
A judge correlating at 0.1 is not measuring what you think it is, and no
amount of tuning the threshold will help.

**`spearman` above `pearson`** means the judge ranks correctly but its
scale is compressed - it puts everything between 0.4 and 0.6. That is
usually fine, and usually fixed by scoring against a rubric rather than
asking for a number.

**`kappa`** is the honest version of `agreement`. Two raters who mark
everything as a pass agree completely and have told you nothing; kappa
reports that as `None` rather than as 1.0. Roughly: below 0.4 is poor,
above 0.6 is usable.

## The blunt verdict

```python
report.trustworthy      # True, False, or None
```

`True` when rank correlation is at least 0.6 over at least twenty pairs.
It abstains with `None` below that, rather than dressing up noise as a
verdict - which is also why the printed report warns you when the sample
is too small.

## Missing labels

Rows where either side is missing are dropped, not treated as zero, and
counted so the omission stays visible:

```python
report.pairs      # 38
report.dropped    # 2
```

## Picking a metric

A run that scored several metrics needs to know which one your labels
are for:

```python
report = align(result, labels, metric="Faithfulness")
report = align(result, labels, metric=faithfulness)     # same thing
```

With one metric it is unambiguous and you can leave it out. You can also
skip the run entirely and pass scores directly:

```python
report = align([0.9, 0.4, 0.8], [1.0, 0.0, 1.0])
```

## How many rows?

Thirty to fifty, labelled once, is enough to catch a judge that is badly
wrong - which is the failure worth catching. Below twenty, correlation
is noise wearing a number, and `trustworthy` will tell you so.
