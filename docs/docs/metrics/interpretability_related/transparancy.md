# Transparancy

This measure the Transparancy of your RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import transparancy

data = load_data()
result = evaluate(
    data,
    metrics=[
        transparancy,
    ],
)
result
```