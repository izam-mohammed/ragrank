# Tonality

This measure the tonality of the text in RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import tonality

data = load_data()
result = evaluate(
    data,
    metrics=[
        tonality(on="response", tone="doctor"),
    ],
)
result
```