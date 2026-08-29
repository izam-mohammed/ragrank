(the-metrics)=
# 🧩 Metrics

A metric answers one question about one datapoint and returns a score
between 0 and 1. Ragrank ships two kinds, and the difference matters
more than any individual metric does.

## Judged metrics cost money

These ask a language model. They handle questions no amount of string
comparison can, and every call has a price and some variance.

| Metric | The question it asks |
| --- | --- |
| [`faithfulness`](./response_related/faithfulness.md) | Is the model making things up? |
| [`correctness`](./response_related/correctness.md) | Is the answer right, against a known one? |
| [`response_relevancy`](./response_related/response_relevancy.md) | Does the answer address the question? |
| [`response_conciseness`](./response_related/response_conciseness.md) | Or does it waffle? |
| [`context_relevancy`](./context_related/context_relevancy.md) | Did retrieval return anything useful? |
| [`context_utilization`](./context_related/context_utilization.md) | Did the model use what it was given? |

## Free metrics cost nothing

These are computed in Python. No model, no network, no variance, and
the same answer every time.

| Group | Metrics | Needs |
| --- | --- | --- |
| [Retrieval](./retrieval_metrics/index.md) | `hit_rate`, `mrr`, `precision_at_k`, `recall_at_k`, `ndcg`, `mean_average_precision` | `retrieved_ids` and `reference_ids` |
| [Text](./heuristic_metrics/index.md) | `exact_match`, `token_f1`, `rouge_l`, `levenshtein_ratio`, `string_presence` | `reference` |
| [Semantic](./heuristic_metrics/index.md#semantic-similarity) | `semantic_similarity` | `reference` and an embedding model |
| [Format](./heuristic_metrics/index.md#format-checking) | `json_valid` | nothing |

## Start with the free ones

This is the advice most likely to save you money.

If retrieval is broken, `hit_rate = 0.31` tells you more than any
language model's opinion of your context, and it costs nothing to find
out. Diagnose the cheap way first, then spend on judged metrics for the
questions that genuinely need judgement.

```python
from ragrank import evaluate
from ragrank.metric import RETRIEVAL_METRICS

evaluate(data, metrics=RETRIEVAL_METRICS)
```

## Presets

```python
from ragrank.metric import RAG_TRIAD, RETRIEVAL_METRICS
```

`RAG_TRIAD` is `context_relevancy`, `faithfulness` and
`response_relevancy`. One leg per failure mode: bad retrieval, an
ungrounded answer, and an answer that misses the point. Between them
they tell you *where* a pipeline broke, not just that it did.

`RETRIEVAL_METRICS` is the free ranking set.

## Every metric can abstain

A metric that cannot score a row returns `None` with an `error`, rather
than inventing a zero. A response with no factual claims is not
unfaithful, and a row with no reference cannot be judged for
correctness. Those are different from scoring badly, and the result
says so.

```python
result = evaluate(data, metrics=[faithfulness])
result.failed_count          # how many (row, metric) pairs had no score
result.results[0][0].error   # and why
```

```{Warning}
The metrics in Ragrank are still under development and research. If you
find any errors while scoring, please raise an [issue on
GitHub](https://github.com/izam-mohammed/ragrank/issues). We are
actively working to improve them, and none of them yet ship with
published benchmark numbers.
```

```{toctree}
:hidden:

response_related/index
context_related/index
retrieval_metrics/index
heuristic_metrics/index
custom_metrics/index
```
