# Informativeness

Measures the informativeness of the RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import informativeness

data = load_data()
result = evaluate(
    data,
    metrics=[
        informativeness,
    ],
)
result
```