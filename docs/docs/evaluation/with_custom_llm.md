# with custom LLM

You can use different LLM in evaluation. The default one is OpenAI

```python
from ragrank import evaluate
from ragrank.dataset import Dataset
from ragrank.llm import vertexai

data = Dataset()
llm = vertexai()

result = evaluate(data, llm=llm)
result
```