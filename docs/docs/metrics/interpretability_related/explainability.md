# Explainability

This measure the explainability of the RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import explainability

data = load_data()
result = evaluate(
    data,
    metrics=[
        explainability,
    ],
)
result
```