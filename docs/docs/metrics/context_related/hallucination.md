# Hallucination

This measure the hallucination score for the context of the model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import hallucination

data = load_data()
result = evaluate(
    data,
    metrics=[
        hallucination,
    ],
)
result
```