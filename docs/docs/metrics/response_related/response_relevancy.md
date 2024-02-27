# Response Relevancy

Measures the Response Relevance of the RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import response_relevancy

data = load_data()
result = evaluate(
    data,
    metrics=[
        response_relevancy,
    ],
)
result
```