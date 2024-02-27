# Coherence

Measures the coherence of the RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import coherence

data = load_data()
result = evaluate(
    data,
    metrics=[
        coherence,
    ],
)
result
```