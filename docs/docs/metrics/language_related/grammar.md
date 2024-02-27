# Grammar

This measure the grammar of the text in RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import grammar

data = load_data()
result = evaluate(
    data,
    metrics=[
        grammar(on="response"),
    ],
)
result
```