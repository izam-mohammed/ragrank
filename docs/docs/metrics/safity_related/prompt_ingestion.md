# Prompt Ingestion

Measures the Prompt Ingestion of the RAG model

```python 
from ragrank import evaluate
from ragrank.dataset import load_data
from ragrank.metrics import prompt_ingestion

data = load_data()
result = evaluate(
    data,
    metrics=[
        prompt_ingestion,
    ],
)
result
```