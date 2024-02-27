# Creativity

Measures the creativity of the RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import creativity

data = load_data()
result = evaluate(
    data,
    metrics=[
        creativity,
    ],
)
result
```