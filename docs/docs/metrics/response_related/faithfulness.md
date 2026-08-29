(faithfulness)=
# Faithfulness

Faithfulness answers the question most people come to a RAG evaluation
library for: **is the model making things up?**

A low score means the generator produced content the retrieved context
does not support, regardless of whether that content happens to be true
in the world. That is the failure mode that makes a RAG system
untrustworthy, and it is different from retrieving the wrong documents.

```python
from ragrank import evaluate
from ragrank.dataset import from_dict
from ragrank.metric import faithfulness

data = from_dict({
    "question": "Where is the Eiffel Tower?",
    "context": ["The Eiffel Tower stands on the Champ de Mars in Paris."],
    "response": "The Eiffel Tower is in Paris. It was built in 1750.",
})

result = evaluate(data, metrics=[faithfulness])
```

## How it works

Rather than asking a model for one verdict on the whole answer,
faithfulness decomposes the response and checks the parts:

1. Split the response into atomic claims, each one independently
   checkable.
2. Verify each claim against the retrieved context.
3. Score the ratio of supported claims to verified claims.

That costs more calls than a single judgement, and it buys something a
single number cannot give you: a score you can point at.

```python
result.results[0][0].metadata["claims"]
# [{"claim": "The Eiffel Tower is in Paris.", "supported": 1.0},
#  {"claim": "It was built in 1750.",         "supported": 0.0}]
```

A score of 0.5 with that attached tells you *which sentence* was
invented. A score of 0.5 on its own tells you almost nothing.

## What it needs

`context` and `response`. No reference answer, because this is a
grounding check rather than a correctness check. For correctness see
[`correctness`](./correctness.md); the two ask genuinely different
questions and a response can be grounded and wrong, or right and
ungrounded.

## When it abstains

Faithfulness returns `None` rather than a score in two cases, and
neither is a failure:

- **The response makes no factual claims.** "Thanks for asking!" is not
  unfaithful, there is simply nothing to verify.
- **There is no context.** Nothing to verify against, so no call is
  made at all and nothing is spent.

## Controlling the cost

Each row costs one extraction call plus one call per claim, so a long
response is an expensive one. `max_claims` caps that:

```python
from ragrank.metric import Faithfulness

cheaper = Faithfulness(max_claims=10)
```

The default is 50. Verified and total claim counts are reported in
`metadata`, so you can see what you paid for:

```python
result.results[0][0].metadata["claim_count"]
result.results[0][0].metadata["verified_count"]
```

## Building on it

`Faithfulness` is a thin layer over `ClaimMetric`, which is a reusable
primitive rather than a single metric. Decomposing a text into claims
and verifying them against a source is the expensive part, and several
further metrics can be derived from the same pass. Subclass it by
saying which text to decompose and which text to check it against.
