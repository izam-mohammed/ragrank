(basic-evaluation)=
# Basic Evaluation


Set your `OPENAI_API_KEY` as an environment variable

```{admonition} Attention
:class: Tip

By default, we are using the OpenAI LLM for internal operations. You can change it later on. So please set your valid `OPENAI_API_KEY`, otherwise you will get internal error 🤓.
```

```bash
export OPENAI_API_KEY="..."
```

Performing the evaluation in python

```python
from ragrank import evaluate
from ragrank.dataset import from_dict
from ragrank.metric import response_relevancy

# Define your dataset
data = from_dict({
    "question": "What is the capital of France?",
    "context": ["France is famous for its iconic landmarks such as the Eiffel Tower and its rich culinary tradition."],
    "answer": "The capital of France is Paris.",
})

# Evaluate the response relevance metric
result = evaluate(data=data, metrics=[response_relevancy])

# Display the evaluation results
result.to_dataframe()
```

Congratulations 🎉, you have done your first step. 
> A journey of thousand miles starts with the first step 🌱.


Now you can deep dive into the [core concepts 🔥](../core_concepts/index.md) of RAG evaluation.