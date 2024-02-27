# with custom embedding

We can use custom embedding methods if we wanted to.

```python
from ragrank.embedding import openai_embedding
from ragrank.dataset import Dataset
from ragrank import evaluate

data = Dataset()
embedding = openai_embedding()
results = evaluate(data, embedding=embedding)
results
```