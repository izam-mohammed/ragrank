# Transparency

This measure the Transparency of your RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import transparency

data = load_data()
result = evaluate(
    data,
    metrics=[
        transparency,
    ],
)
result
```