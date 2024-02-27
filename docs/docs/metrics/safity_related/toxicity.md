# Toxicity

Measures the Toxicity of the RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import toxicity

data = load_data()
result = evaluate(
    data,
    metrics=[
        toxicity,
    ],
)
result
```