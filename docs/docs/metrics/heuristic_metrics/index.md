(heuristic-metrics)=
# ⚡ Free Metrics

Metrics computed in Python. No language model, no network, no
credentials, and the same answer every time. They run in microseconds
and cost nothing, which makes them the right first choice for anything
they can actually answer.

```python
from ragrank import evaluate
from ragrank.metric import exact_match, token_f1, rouge_l

evaluate(data, metrics=[exact_match, token_f1, rouge_l])
```

## Text comparison

All of these compare `response` against `reference`.

| Metric | What it measures |
| --- | --- |
| `exact_match` | Identical after normalisation |
| `string_presence` | The reference appears somewhere in the response |
| `levenshtein_ratio` | Character level similarity |
| `token_f1` | Token overlap, forgiving about word order |
| `rouge_l` | Longest common subsequence, sensitive to order |

### Normalisation

Comparison is not literal. Text is lowercased, punctuation and articles
are stripped, and runs of whitespace are collapsed, so `"The Paris."`
and `"paris"` compare equal. This is the standard SQuAD normalisation.

### Choosing between them

`token_f1` ignores word order, so a reordered answer still scores well.
`rouge_l` is order sensitive, so it rewards phrasing that follows the
reference. `exact_match` is a blunt pass or fail and is the right
choice when there genuinely is only one acceptable answer.

(semantic-similarity)=
## Semantic similarity

Sits between token overlap and a judged metric. Cheaper and faster than
asking a model, far more forgiving than comparing strings.

```python
from ragrank.metric import SemanticSimilarity
from ragrank.embedding import BaseEmbedding

metric = SemanticSimilarity(embedding=my_embedding_model)
```

No default embedding model ships with Ragrank, because which model to
embed with is a deployment decision. A metric used without one returns
`None` and an error saying exactly what to pass, rather than silently
scoring nothing.

Implement `BaseEmbedding` to plug in a provider:

```python
from ragrank.embedding import BaseEmbedding

class MyEmbedding(BaseEmbedding):
    @property
    def name(self) -> str:
        return "my-model"

    def embed_text(self, text: str) -> list[float]:
        return my_provider.embed(text)
```

`FakeEmbedding` is included for testing. It derives stable vectors from
a hash, so the wiring can be exercised with no provider, but it carries
no real semantics and its scores mean nothing.

(json-valid)=
## Format checking

`json_valid` scores 1.0 when the response parses as JSON and 0.0 when
it does not. It needs only `response`, no reference.

```python
from ragrank.metric import json_valid

evaluate(data, metrics=[json_valid])
```

## Abstaining

Every reference-based metric here returns `None` when the row has no
reference, rather than inventing a zero. A missing ground truth is not
the same as a wrong answer, and the result says which one it was.
