# Detailed Evaluation

```{note}
Before evaluating the RAG, make sure to record all of the questions, context and the generated answer from the RAG model.
```

We can load the data first. The data should have these keys

- `question`: The question you want to ask
- `context`: The context relevant to the question
- `response`: The response to the question

```{hint}
If your data columns in another name, you can use the `ColumnMap` in dataset. See [Data Ingestion](./data_ingestion.md) to learn more.
```

```python
from ragrank.dataset import from_dict

data = from_dict(
    {
        "question": "What is the capital of France?",
        "context": [
            "France is famous for its iconic landmarks such as the Eiffel Tower and its rich culinary tradition."
        ],
        "response": "The capital of France is Paris.",
    },
    return_as_dataset=True,
)
```

```{seealso}
You can ingestion data in multiple ways such as from a csv file or from a Pandas Dataframe. Check the [Data Ingestion](./data_ingestion.md) for more.
```

Now We have to set the LLM and other configuration

```python
from ragrank.llm import default_llm
```

```{Note}
By default Ragrank is using Openai LLM for internal operations. You can change this LLM as what ever you want. Checkout the [LLM Integrations](./with_llm.md) for more.
```

Getting the evaluation metrics from ragrank
```python
from ragrank.metric import (
    response_relevancy,
    response_conciseness,
    context_relevancy,
    context_utilization,
)
```

Now that we have our data, we can evaluate it using Ragrank. We use the `evaluate` method to do this. This method takes the following arguments:

- `dataset`: The data you want to evaluate upon
- `llm`: LLM that internally use for evaluation.
- `metrics`: The evaluations you want to perform on your data

```python
from ragrank import evaluate
from ragrank.evaluation import EvalResult

result:EvalResult = evaluate(
    dataset=data,
    llm=default_llm(),
    metrics=[
        response_relevancy,
        response_conciseness,
        context_relevancy,
        context_utilization,
    ],
)
```

Show the result data.

```python
print(result)
```