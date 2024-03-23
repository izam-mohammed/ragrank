(with-custom-llm)=
# With LLMs

You can use different LLM in evaluation. The default one is OpenAI. In addition to that You can use `LLMWrappers` as LLMs

```python
from ragrank.llm import LLMConfig
from ragrank.integrations.llm import OpenaiLLM
from ragrank.dataset import DataNode
from ragrank import evaluate

# defining the data node
data = DataNode(
    question="What is the tallest mountain in the world?",
    context=[
        "Mount Everest is the tallest mountain above sea level.",
        "It is located in the Himalayas.",
    ],
    response="The tallest mountain in the world is Mount Everest.",
)

llm_configuration = LLMConfig(
    temperature=0.1,
    max_tokens=200,
    seed=1,
    top_p=0.9,
    stop=None
)
llm = OpenaiLLM(llm_config=llm_configuration)

result = evaluate(
    dataset=data,
    llm=llm,
)

print(result)
```

## LLM Wrapper

There are numerous LLM providers like Langchain and Llama-index. To seamlessly integrate with their LLM classes, we introduce Ragrank LLM wrappers. These wrappers serve as a bridge between their LLM classes and Ragrank's LLM classes.

Here is an example of **Langchain LLM wrapper**:
```python
from langchain_openai import OpenAI

# we are using langchain openai llm for ragrank evaluation
langchain_llm = OpenAI()
```

If we pass this LLM to the `evaluate` method, it will throw an error. Instead, wrap this LLM in a wrapper

```python
from ragrank.integrations.llm import LangchainLLMWrapper

ragrank_llm = LangchainLLMWrapper(langchain_llm)
```

Now we can use this wrapped `ragrank_llm` for evaluation

```python
from ragrank.dataset import DataNode
from ragrank import evaluate

data = DataNode(
    question="What is the tallest mountain in the world?",
    context=[
        "Mount Everest is the tallest mountain above sea level.",
        "It is located in the Himalayas.",
    ],
    response="The tallest mountain in the world is Mount Everest.",
)

result = evaluate(
    dataset = data,
    llm = ragrank_llm,
)
```

## LLM Config


We can configure the LLM for its internal operations. 

Available configurations include:
- `temperature` : Controls the randomness of generated text. Higher values result in more diverse outputs.
- `max_tokens` : Specifies the maximum number of tokens (words or subwords) in the generated text.
- `seed` : Sets the initial seed for random number generation, ensuring reproducibility of results.
- `top_p` : Also known as nucleus sampling, it sets the probability threshold for sampling from the most likely tokens.
- `stop` : Specifies whether to include a stop token to halt generation at a certain point.

```python
from ragrank.llm import LLMConfig

llm_configuration = LLMConfig(
    temperature=0.1,
    max_tokens=200,
    seed=1,
    top_p=0.9,
    stop=None
)
```