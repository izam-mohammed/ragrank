# Generalizability

Measures the Generalizability of the RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import generalizability

data = load_data()
result = evaluate(
    data,
    metrics=[
        generalizability,
    ],
)
result
```