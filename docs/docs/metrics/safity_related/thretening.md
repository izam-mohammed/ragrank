# Thretening

Measures the Thretening of the RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import thretening

data = load_data()
result = evaluate(
    data,
    metrics=[
        thretening,
    ],
)
result
```