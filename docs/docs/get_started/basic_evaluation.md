(basic-evaluation)=
# Basic Evaluation

here is the code for basic evaluation using `ragrank`

```python
from ragrank import evaluate, Dataset

data = Dataset({
    "question": ["Who is the president of America ?",],
    "context": [["The president of America has go to India",],],
    "answer": ["I don't know"],
})

result = evaluate(data)
result.to_dataframe()
```

Congratulations, you have done your first step.