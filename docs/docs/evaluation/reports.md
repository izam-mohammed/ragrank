(html-reports)=
# 📄 HTML reports

A DataFrame is the right shape for whoever is holding the notebook. It
is the wrong shape for the person who asked whether the change is safe
to ship.

```python
result.to_html("report.html")
```

One self-contained file. No assets, no scripts, no network, nothing to
serve. Open it, or attach it to a pull request.

## What is in it

- A header: rows, metrics, which judge, how long, how many tokens, and
  the overall verdict if any metric declared a threshold
- Per metric: the score with its standard error, how many rows scored,
  the pass rate, the verdict, and the metric's cost tier
- Per row: the question, the response, and every metric's score, with
  the judge's reasoning or the parse error folded into a disclosure

Everything from the dataset is escaped. A response is untrusted text,
and a report that renders it as markup is a report you cannot hand to
anyone.

## From the command line

```bash
ragrank eval ragrank.yaml --html report.html
```

Which is where it earns its keep: a CI job that fails a threshold can
leave behind the page explaining which rows caused it.

## From a test suite

The pytest plugin collects every evaluation in a session into one
document:

```bash
pytest -m ragrank --ragrank-report=evals.html
```

Each test becomes its own section, named by its node id. See
[Testing](../testing/index.md) for the fixtures that feed it.

## A title

```python
result.to_html("nightly.html", title="nightly eval, main")
```

The document is returned as a string as well as written, so you can post
it somewhere instead of writing it to disk:

```python
html = result.to_html()
```
