(correctness)=
# Correctness

Correctness asks whether the answer is **right**, judged against a
reference answer you already know.

```python
from ragrank import evaluate
from ragrank.dataset import from_dict
from ragrank.metric import correctness

data = from_dict({
    "question": "What is the capital of France?",
    "context": ["France is in Europe."],
    "response": "The capital of France is Paris.",
    "reference": "Paris",
})

evaluate(data, metrics=[correctness])
```

## Why not just compare the strings

Because the strings do not match. `exact_match` scores that example
0.0, and `token_f1` scores it poorly, even though the answer is
perfectly correct. Wording, length and formatting are not the thing you
wanted to measure.

| Metric | "The capital of France is Paris." vs "Paris" |
| --- | --- |
| `exact_match` | 0.0 |
| `token_f1` | low |
| `semantic_similarity` | high, and free |
| `correctness` | 1.0, and costs a call |

Reach for the cheap metrics first. Use `correctness` when the answers
are open enough that surface comparison genuinely misleads you.

## The scale

The judge picks one of three verdicts rather than producing a number:

| Verdict | Meaning | Score |
| --- | --- | --- |
| A | Says the same thing as the reference, or contains it with consistent extra detail | 1.0 |
| B | Partly right, omits or garbles part of the reference | 0.5 |
| C | Disagrees with the reference, or does not answer | 0.0 |

Asking for a letter and mapping it to a number in Python is markedly
more reliable than asking a model for a float. An answer outside the
rubric is rejected rather than coerced, so the row abstains instead of
producing a made-up score.

## What it needs

`question`, `reference` and `response`. Without a reference the metric
cannot apply, and the evaluation run will tell you so before spending
anything.
