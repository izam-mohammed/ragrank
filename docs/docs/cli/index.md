(the-cli)=
# 💻 Command Line

An evaluation described by a file is diffable, reviewable in a pull
request, and editable by somebody who does not write Python.

```bash
ragrank eval ragrank.yaml
```

## A config

```yaml
dataset: data.csv

metrics:
  - faithfulness
  - context_relevancy
  - name: token_f1
    threshold: 0.9

run:
  max_workers: 8
  cache: true
```

Or inline, with no CSV:

```yaml
data:
  question: ["What is the capital of France?"]
  context: [["Paris is the capital of France."]]
  response: ["Paris"]
  reference: ["Paris"]

metrics:
  - exact_match
```

YAML needs PyYAML (`pip install pyyaml`), which is deliberately not a
dependency of Ragrank. A `.json` config works with nothing extra
installed.

## Exit codes are the CI contract

| Code | Meaning |
| --- | --- |
| 0 | Everything passed |
| 1 | A metric fell below its threshold |
| 2 | The config was wrong |

```bash
ragrank eval ragrank.yaml || echo "evaluation failed"
```

A metric with a `threshold` in the config gates the build, so gating
needs no Python at all.

## Saving and comparing runs

```bash
ragrank eval config.json --output before.json
# change a prompt, a model, a chunking strategy
ragrank eval config.json --output after.json

ragrank compare before.json after.json
```

```text
Faithfulness: 0.812 -> 0.901 (+0.089)
Token F1: 0.640 -> 0.638 (-0.002)
```

## An HTML report

```bash
ragrank eval ragrank.yaml --html report.html
```

One self-contained page with every row, every score and the judge's
reasoning - the useful artefact for a CI job that just failed a
threshold. See [HTML reports](../evaluation/reports.md).

## Column mapping

If your CSV uses different column names:

```yaml
dataset: data.csv
column_map:
  question: query
  response: answer
metrics:
  - exact_match
```

## Errors are actionable

```text
ERROR: Unknown metric 'faithfullness'. Available: context_relevancy,
context_utilization, correctness, exact_match, faithfulness, ...
```
