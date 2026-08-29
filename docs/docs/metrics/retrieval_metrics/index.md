(retrieval-metrics)=
# 🔍 Retrieval Metrics

The classic information retrieval measures, computed from document ids.
No language model, no cost, no variance, and the same answer every
time.

**These are the first thing to run.** If your retriever is not finding
the right documents, no judgement of the generated answer will explain
why, and `hit_rate = 0.31` is a far sharper signal than any model's
opinion of your context.

```python
from ragrank import evaluate
from ragrank.dataset import from_dict
from ragrank.metric import RETRIEVAL_METRICS

data = from_dict({
    "question": ["Where is the Eiffel Tower?"],
    "context": [["The Eiffel Tower is in Paris."]],
    "response": ["Paris"],
    "retrieved_ids": [["doc_7", "doc_1", "doc_3"]],
    "reference_ids": [["doc_1"]],
}, return_as_dataset=True)

evaluate(data, metrics=RETRIEVAL_METRICS)
```

## The metrics

| Metric | What it measures |
| --- | --- |
| `hit_rate` | Did retrieval return at least one relevant document? |
| `mrr` | How high up was the first relevant document? |
| `precision_at_k` | What fraction of what came back was relevant? |
| `recall_at_k` | What fraction of the relevant documents came back? |
| `ndcg` | Rank-weighted relevance, discounted by position |
| `mean_average_precision` | Precision at each hit, averaged |

## Cutting off at k

Every one of them takes an optional `k`, which considers only the top
k results. The name reflects it, so several cutoffs can appear in one
report without colliding:

```python
from ragrank.metric import RecallAtK, HitRate

evaluate(data, metrics=[RecallAtK(k=1), RecallAtK(k=5), HitRate(k=3)])
# Recall@1, Recall@5, Hit Rate@3
```

## What they need

`retrieved_ids` and `reference_ids` on the data. These are the ids your
retriever returned, in rank order, and the ids that should have been
returned.

```python
from ragrank.dataset import DataNode

DataNode(
    question="...",
    context=["..."],
    response="...",
    retrieved_ids=["doc_7", "doc_1"],   # in rank order
    reference_ids=["doc_1"],            # what should have come back
)
```

If they are absent, the run stops before spending anything and names
the metric and the missing column.

## Reading the numbers

The three that matter most, and what a low one means:

- **Low `hit_rate`** means the right document is not being retrieved at
  all. Look at chunking and the embedding model, not the prompt.
- **Low `mrr` with a high `hit_rate`** means the right document is
  found but ranked badly. Look at reranking.
- **Low `precision` with a high `recall`** means you are retrieving too
  much. Look at your `k`.
