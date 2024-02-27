# Originality

Measures the originality of the RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import originality

data = load_data()
result = evaluate(
    data,
    metrics=[
        originality,
    ],
)
result
```