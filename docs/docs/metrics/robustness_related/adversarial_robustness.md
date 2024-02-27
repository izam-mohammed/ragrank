# Adversarial Robustness

Measures the Adversarial Robustness of the RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import adversarial_robustness

data = load_data()
result = evaluate(
    data,
    metrics=[
        adversarial_robustness,
    ],
)
result
```