# Bias

Measures the Bias of the RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import bias

data = load_data()
result = evaluate(
    data,
    metrics=[
        bias,
    ],
)
result
```