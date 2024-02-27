# Fluency

This measure the Fluency of the text in RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import fluency

data = load_data()
result = evaluate(
    data,
    metrics=[
        fluency(on="response"),
    ],
)
result
```