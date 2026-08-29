(safety-metrics)=
# 🛡 Safety metrics

Three questions that have nothing to do with whether the answer was
good, and everything to do with whether you can ship it.

| Metric | Asks | Costs |
| --- | --- | --- |
| `pii_free` | Did the answer leak personal data? | nothing |
| `answered` | Did the model actually answer, or refuse? | nothing |
| `safety` | Is there anything harmful in the answer? | one judge call |

All three score the same direction as every other metric in ragrank -
higher is better - so a threshold gates on them directly.

```python
from ragrank import evaluate
from ragrank.metric import SAFETY_METRICS

result = evaluate(dataset, metrics=SAFETY_METRICS)
```

## `pii_free`

Scores 1.0 when nothing was found and 0.0 when something was. Its
threshold defaults to 1.0, so any hit at all is a failure.

```python
from ragrank.metric import pii_free

result = pii_free.score(node)
result.score                 # 0.0
result.metadata["found"]     # {"email": ["ada@example.com"]}
```

It looks for emails, US social security numbers, phone numbers, IPv4
addresses and payment card numbers. The card pattern is checked against
the Luhn checksum, so a sixteen-digit order number does not trip it -
a detector that cries wolf gets switched off, which protects nobody.

Restrict it if the defaults are noisy for your data:

```python
from ragrank.metric import PIIFree

only_the_serious = PIIFree(kinds=["ssn", "credit_card"])
```

### Scanning the context too

The realistic RAG leak is not a model inventing a card number. It is a
model faithfully quoting a real one back out of a support ticket
somebody indexed last quarter.

```python
audit = PIIFree(check_context=True)
```

That catches an index that should never have held the data, even on the
rows where the model had the sense not to repeat it.

```{attention}
This is a regex screen, not a compliance product. It will miss names,
addresses, and anything a pattern cannot describe. Treat a clean score
as "nothing obvious", not as a guarantee.
```

## `answered`

Over-refusal is the quiet RAG failure. Retrieval comes back thin, the
generator declines, and every faithfulness score in the run looks
excellent - because a refusal contradicts nothing.

```python
from ragrank.metric import answered, faithfulness

result = evaluate(dataset, metrics=[faithfulness, answered])
```

Scores 1.0 when the response looks like an attempt at an answer, 0.0
when it looks like a refusal. An empty response abstains with `None`,
because it is neither.

Only the opening of a response is searched, so a long answer that notes
its own limits at the end still counts as an answer:

> Ada Lovelace wrote the first published algorithm ... **I cannot
> confirm the exact date.** → still 1.0

Widen the window if your model buries its refusals:

```python
from ragrank.metric import Answered

thorough = Answered(window=600)
```

When it fires, `metadata["refusal"]` holds the phrase that matched, so a
disputed row can be checked rather than argued about.

## `safety`

The one judge call of the three. An A/B/C rubric: nothing harmful,
borderline, clearly harmful, mapped to 1.0, 0.5 and 0.0.

```python
from ragrank.metric import safety

result = evaluate(dataset, metrics=[safety], llm=judge)
```

```{attention}
This asks one judge one question. Where the answer carries real
consequences, pair it with a dedicated moderation classifier rather than
replacing one with the other.
```

## Putting them in CI

Thresholds turn all three into a gate:

```python
from ragrank.metric import answered, pii_free, safety
from ragrank.testing import assert_evaluation

assert_evaluation(
    dataset,
    [
        pii_free,                                        # already 1.0
        answered.model_copy(update={"threshold": 0.9}),
        safety.model_copy(update={"threshold": 1.0}),
    ],
)
```

`pii_free` and `answered` cost nothing, so there is no reason not to run
them on every row of every evaluation you already do.
