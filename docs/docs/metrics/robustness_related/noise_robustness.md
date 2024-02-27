# Noise Robustness

Measures the Noise Robustness of the RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import noise_robustness

data = load_data()
result = evaluate(
    data,
    metrics=[
        noise_robustness,
    ],
)
result
```