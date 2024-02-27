# Response Currectness

Measures the Response Currectness of the RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import response_currectness

data = load_data()
result = evaluate(
    data,
    metrics=[
        response_currectness,
    ],
)
result
```