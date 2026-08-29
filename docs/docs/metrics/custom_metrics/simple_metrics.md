(simple-metrics)=
# Metrics in One Expression

Three ways to define a metric, none of which need a class.

## Any Python you like

```python
from ragrank.metric import metric

@metric(name="Has citation", threshold=1.0)
def has_citation(response: str) -> bool:
    return "[" in response
```

Parameters are injected by name from the `DataNode`, so a function
declares exactly what it needs and gets nothing else:

```python
@metric
def length_ratio(response: str, reference: str) -> float:
    return min(len(response) / max(len(reference), 1), 1.0)
```

That also gives the evaluation runner the metric's required columns for
free, so a dataset missing `reference` fails immediately rather than
part way through a paid run.

A parameter that is not a `DataNode` field is rejected when the metric
is defined, not when it is used:

```python
@metric
def broken(nonsense: str) -> float:
    return 1.0
# ValueError: broken() asks for ['nonsense'], which are not DataNode fields.
```

Returning `None` means the metric does not apply to that row, which is
recorded honestly rather than as a zero.

## A judge with a rubric

```python
from ragrank.metric import LLMJudge

tone = LLMJudge(
    judge_name="Tone",
    instructions="Is the response's tone right for a support agent?",
    rubric={"A": 1.0, "B": 0.5, "C": 0.0},
)
```

The model picks a label and the numbers stay in Python. This is
deliberate: asking a model for a letter is markedly more reliable than
asking it for a float, and an answer outside the rubric is rejected
rather than coerced into a misleading score.

Few-shot examples calibrate it:

```python
tone = LLMJudge(
    judge_name="Tone",
    instructions="...",
    rubric={"A": 1.0, "B": 0.0},
    examples=[{"response": "Sure, happy to help.", "verdict": "A"}],
)
```

## A rule in plain English

```python
from ragrank.metric import Guidelines

policy = Guidelines(
    judge_name="No medical advice",
    guidelines="The response must never give medical advice.",
)
```

Binary, and gates by default with a threshold of 1.0. This covers a
surprising amount of what people actually want to measure, with no
scaffolding at all.
