> still under development ~ the api moves around a bit

<p align="center">
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Auto-Playground/Ragrank/main/docs/docs/_static/imgs/ragrank_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Auto-Playground/Ragrank/main/docs/docs/_static/imgs/ragrank_light.png">
    <img alt="ragrank" src="https://raw.githubusercontent.com/Auto-Playground/Ragrank/main/docs/docs/_static/imgs/ragrank_light.png" height="130">
    </picture>
</p>

<p align="center">
    <a href="">
        <img alt="license" src="https://img.shields.io/github/license/Auto-Playground/ragrank">
    </a>
    <a href="https://pypi.org/project/ragrank/">
        <img alt="python versions" src="https://img.shields.io/pypi/pyversions/ragrank">
    </a>
    <a href="https://ragrank.readthedocs.io/latest/">
        <img alt="docs" src="https://img.shields.io/readthedocs/ragrank">
    </a>
    <a href="https://pypi.org/project/ragrank/">
        <img alt="release" src="https://img.shields.io/github/v/release/Auto-Playground/Ragrank?color=orange">
    </a>
    <a href="https://github.com/Auto-Playground/Ragrank/actions">
        <img alt="tests" src="https://img.shields.io/github/actions/workflow/status/Auto-Playground/ragrank/.github%2Fworkflows%2Ftests.yml">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://ragrank.readthedocs.io/latest/">Documentation</a> |
        <a href="https://api-ragrank.readthedocs.io/">API reference</a> |
        <a href="https://ragrank.readthedocs.io/latest/get_started/basic_evaluation.html">Quickstart</a> |
        <a href="https://discord.gg/zDzM5hFS">Discord</a> |
        <a href="https://pypi.org/project/ragrank/">PyPI</a>
    <p>
</h4>

ragrank scores your RAG pipeline so you don't have to squint at outputs and go "yeah, that
looks about right". you hand it questions, retrieved context and responses ~ it hands back
numbers you can put in CI.

## install

```bash
pip install ragrank
```

that's everything you need to evaluate. extras are only for provider and
framework sdks ~ the stuff not everyone wants installed:

```bash
pip install "ragrank[openai]"       # the openai judge
pip install "ragrank[langchain]"    # LangchainLLMWrapper
pip install "ragrank[llama-index]"  # LlamaindexLLMWrapper
pip install "ragrank[hf]"           # from_hfdataset()
pip install "ragrank[all]"          # all of them
```

from source:

```bash
git clone https://github.com/Auto-Playground/ragrank.git && cd ragrank
uv sync --group dev
```

## quick start

```bash
export OPENAI_API_KEY="..."
```

```python
from ragrank import evaluate
from ragrank.dataset import from_dict
from ragrank.metric import response_relevancy

data = from_dict({
    "question": "What is the capital of France?",
    "context": ["France is famous for the Eiffel Tower and its food."],
    "response": "The capital of France is Paris.",
})

result = evaluate(data, metrics=[response_relevancy])
print(result)
```

```
Response Relevancy: 0.850
```

`result.to_dataframe()` if you want a table, `result.to_json()` if you don't want pandas.

## bring your own model

any model, not just openai. pass it once and every metric uses it:

```python
from ragrank.integrations.langchain import LangchainLLMWrapper
from langchain_community.chat_models import ChatOllama

result = evaluate(data, llm=LangchainLLMWrapper(llm=ChatOllama(model="gemma:2b")))
```

or write your own ~ subclass `BaseLLM`, implement `generate_text`, done.

## no key, no problem

there's a `FakeLLM` in the box, so you can wire up your whole eval pipeline before spending a
cent:

```python
from ragrank.llm import FakeLLM

evaluate(data, llm=FakeLLM(responses=["0.8", "0.3"]))
```

it also takes a `response_fn` if you want the answer to depend on the prompt. handy for tests.

## metrics

| metric | what it's askin' |
| --- | --- |
| `response_relevancy` | does the answer actually answer the question |
| `response_conciseness` | or does it waffle |
| `context_relevancy` | did retrieval pull back anything useful |
| `context_utilization` | did the model bother to read it |

`RAG_TRIAD` is the three that between them tell you *where* it broke:

```python
from ragrank.metric import RAG_TRIAD

evaluate(data, metrics=RAG_TRIAD)
```

roll your own with `CustomInstruct`:

```python
from ragrank.metric import CustomInstruct, InstructConfig, MetricType

politeness = CustomInstruct(config=InstructConfig(
    metric_type=MetricType.NON_BINARY,
    name="Politeness",
    instructions="Score how polite the response is.",
    input_fields=["question", "response"],
))
```

## gating in ci

give a metric a threshold and the result knows whether it passed:

```python
strict = response_relevancy.model_copy(update={"threshold": 0.7})
result = evaluate(dataset, metrics=[strict])

assert result.passed, result
```

## running it properly

`RunConfig` is the one place run behaviour lives:

```python
from ragrank.evaluation import RunConfig

evaluate(dataset, run_config=RunConfig(
    max_workers=8,        # concurrent metric calls
    max_retries=2,        # retries on a failing llm call, with backoff
    show_progress=True,
    raise_on_error=False, # one bad row shouldn't kill a 5000-row run
))
```

**by default a run finishes even when rows fail.** a row the judge fluffs comes back with
`score=None` and an `error` explainin' why, instead of taking down the whole thing forty
minutes and twelve dollars in. flip `raise_on_error=True` if you'd rather it stop.

```python
result.summary()      # mean, stderr, min, max, pass rate, per metric
result.failed_count   # how many (row, metric) pairs came back empty
```

`summary()` reports standard error, not just a mean. a 0.72 over eleven rows with no error
bar isn't really a 0.72.

## data in

```python
from ragrank.dataset import from_dict, from_csv, from_dataframe, from_hfdataset, ColumnMap

from_csv("evals.csv", column_map=ColumnMap(question="query", response="answer"))
```

## development

```bash
make test-offline   # no api key needed
make test           # needs OPENAI_API_KEY
make lint
make format
```

`test-offline` runs the whole thing against `FakeLLM` ~ no network, no spend.

## license

[apache 2.0](https://github.com/Auto-Playground/Ragrank/blob/main/LICENSE). do what you like
with it.

## contributing

issues and PRs welcome. if somethin' is broken, an issue with the traceback is genuinely
useful ~ two of the bugs fixed recently were sitting in the tracker for two years because
nobody re-tested them after they got closed.
