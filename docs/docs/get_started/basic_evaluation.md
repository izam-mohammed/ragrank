(basic-evaluation)=
# Basic Evaluation


Set your `OPENAI_API_KEY` as an environment variable

```{admonition} Attention
:class: Tip

By default, we are using the OpenAI LLM for internal operations. You can change it later on. So please set your valid `OPENAI_API_KEY`, otherwise you will get internal error.
```

Set the API key with running this in the terminal. replace `<your_api_key>` with your actual API key.
```bash
export OPENAI_API_KEY="<your_api_key>"
```

Alternatively, you can set the API key in python with `os` module. replace `<your_api_key>` with your actual API key.

```python
import os

os.environ["OPENAI_API_KEY"] = "<your_api_key>"
```

To perform the evaluation in Python, open the `main.py` file and add the following code:

```python
from ragrank import evaluate
from ragrank.dataset import from_dict
from ragrank.metric import CustomInstruct, InstructConfig

# Define your metric
config = InstructConfig(
    name="Response Relevancy",
    instructions="Evaluate how relevant the response is to the question given the provided context.",
    examples=[],
    input_fields=["question", "context", "response"],
    output_field="relevancy",
)
metric = CustomInstruct(config=config)

# Define your dataset
data = from_dict({
    "question": "What is the capital of France?",
    "context": ["France is famous for its iconic landmarks such as the Eiffel Tower and its rich culinary tradition."],
    "response": "The capital of France is Paris.",
})

# Evaluate
result = evaluate(data, metrics=[metric])

# Display the evaluation results
result.to_dataframe()
```

After adding the code, run the `main.py` file.

Now you can deep dive into the [core concepts](../core_concepts/index.md) of RAG evaluation.
