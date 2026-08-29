(context-reliance)=
# Context reliance

Faithfulness asks whether the answer is grounded. Correctness asks
whether it is right. Neither answers the question a team actually has
after wiring up a retriever:

> Would the model have got this right anyway?

```python
from ragrank import evaluate
from ragrank.metric import context_reliance

result = evaluate(dataset, metrics=[context_reliance])
```

## What it measures

Every claim in the response is checked twice - against the reference, to
see whether it is correct, and against the retrieved context, to see
whether retrieval supplied it. The score is the share of the *correct*
claims that the context supports.

| Score | Reading |
| --- | --- |
| 1.0 | Everything right in the answer came from retrieval |
| 0.5 | Half of it did; the model supplied the rest |
| 0.0 | The model answered from what it already knew |

Claims that are wrong are ignored rather than counted against the
retriever. A hallucination is faithfulness's problem, and folding the
two together gives you a number that moves for two unrelated reasons.

## The interesting case

A low reliance score next to a good correctness score is the finding
worth acting on. The pipeline works, but the model is carrying it, and
it will stop working the moment you point it at documents the model has
never seen - which is to say, your documents.

```python
from ragrank.metric import context_reliance, correctness

result = evaluate(dataset, metrics=[correctness, context_reliance])

# Correctness: 0.910
# Context Reliance: 0.240   <- the retriever is barely contributing
```

RAGChecker calls the gap *self-knowledge*. Ragrank reports the
complement so that higher stays better like everything else here, and
keeps the original direction in metadata:

```python
result.results[0][0].metadata["self_knowledge"]   # 0.76
```

## What it needs

`context`, `response` and `reference`. Without ground truth there is no
way to tell a correct claim from an invented one, so the metric abstains
rather than guessing.

## What it costs

One extraction call, plus up to two verification calls per claim. This
is the most expensive metric in the library. It is a diagnostic to run
once on a sample of fifty rows, not a gate to run on everything.

```python
from ragrank.evaluation import RunConfig

result = evaluate(
    sample,
    metrics=[context_reliance],
    run_config=RunConfig(cache=True),
)
```
