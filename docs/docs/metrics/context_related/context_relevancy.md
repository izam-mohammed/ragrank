# Context relevancy

This measure the context relevance score for the context of the model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import context_relevancy

data = load_data()
result = evaluate(
    data,
    metrics=[
        context_relevancy,
    ],
)
result
```