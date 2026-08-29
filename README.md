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

**judged** ~ these cost an llm call

| metric | what it's askin' |
| --- | --- |
| `faithfulness` | is the model makin' things up |
| `correctness` | is the answer right, against a reference |
| `response_relevancy` | does the answer actually answer the question |
| `response_conciseness` | or does it waffle |
| `context_relevancy` | did retrieval pull back anything useful |
| `context_utilization` | did the model bother to read it |

**free** ~ no llm, no cost, same answer every time

| metric | needs |
| --- | --- |
| `hit_rate` `mrr` `precision_at_k` `recall_at_k` `ndcg` `mean_average_precision` | `retrieved_ids` + `reference_ids` |
| `exact_match` `token_f1` `rouge_l` `levenshtein_ratio` `string_presence` | `reference` |
| `semantic_similarity` | `reference` + an embedding model |
| `json_valid` | nothin' |

if retrieval is broken, `hit_rate=0.31` tells you more than any judge's
opinion of your context ~ and it's free. start there.

`faithfulness` is the one most people want. it splits your answer into
claims and checks each against the context, so a bad score points at the
sentence that caused it:

```python
result.metadata["claims"]
# [{"claim": "The tower is in Paris.",  "supported": 1.0},
#  {"claim": "It was built in 1750.",   "supported": 0.0}]
```

`RAG_TRIAD` is the three that between them tell you *where* it broke, and
`RETRIEVAL_METRICS` is the free ranking set:

```python
from ragrank.metric import RAG_TRIAD, RETRIEVAL_METRICS
evaluate(data, metrics=RAG_TRIAD)
```

## rolling your own

three ways, none of which need a class:

```python
from ragrank.metric import metric, LLMJudge, Guidelines

@metric(name="Has citation", threshold=1.0)
def has_citation(response: str) -> bool:
    return "[" in response

tone = LLMJudge(judge_name="Tone", instructions="Is the tone right for support?",
                rubric={"A": 1.0, "B": 0.5, "C": 0.0})

policy = Guidelines(judge_name="No advice", guidelines="Never give medical advice.")
```

parameters are injected by name from the datanode, so a function asks for
what it needs and gets nothin' else. a parameter that isn't a datanode
field is rejected when you define it, not three hundred rows into a paid
run.

## when one judge isn't enough

judges are noisy. three ways to deal with that:

```python
from ragrank.metric import Jury, Pairwise

# ask several, take the median
Jury(judges=[gpt_judge, claude_judge, local_judge])

# skip absolute scores entirely ~ models are better at "which is better"
Pairwise(baseline_field="reference")

# or just ask the same judge repeatedly and look at the spread
evaluate(data, run_config=RunConfig(repetitions=5, reducer="median"))
```

`Pairwise` judges every pair twice with the order swapped, because judges
favour whatever came first. a verdict that flips is reported as a tie,
not a win.

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
    cache=True,           # reuse identical prompts between runs
    repetitions=1,        # score each row n times and reduce
    raise_on_error=False, # one bad row shouldn't kill a 5000-row run
))
```

**by default a run finishes even when rows fail.** a row the judge fluffs
comes back with `score=None` and an `error` explainin' why, instead of
taking down the whole thing forty minutes and twelve dollars in.

a typo'd option is an error, not a shrug ~ `RunConfig(max_worker=8)`
raises rather than quietly running with the default.

## what it cost you

```python
result.usage
# 240 calls, 61,204 tokens (59,880 in / 1,324 out)

result.cost(per_prompt_token=0.15/1e6, per_response_token=0.60/1e6)
# 0.0097752
```

no price table ships with the library ~ prices change and vary by
provider, so you pass the rates you're actually payin'.

## caching

judges run at temperature 0 over a dataset that barely changes, so addin'
one metric re-asks the other five exactly the same questions:

```python
evaluate(data, run_config=RunConfig(cache=True))
```

one flag for an on-disk cache that survives between runs. off by default
~ no surprise files appear unless you ask.

```
run 1 (cold)       5 calls, 1345 tokens
run 2 (warm)       5 calls, 0 tokens
run 3 (+1 metric)  only the new metric billed
```

## is v2 better than v1

```python
from ragrank.evaluation import compare

diff = compare(baseline, candidate)
print(diff)
# Response Relevancy: 0.517 -> 0.900 (+0.383) [significant]
# Faithfulness:       0.812 -> 0.815 (+0.003) [within noise]

assert not diff.regressed
```

it says when a change is noise, usin' the standard errors the runs already
report. 0.003 is not an improvement, and a library that lets you claim it
is one isn't helpin'. `regressed_rows` points at the datapoints that moved.

## in your test suite

```python
from ragrank.testing import assert_metric

def test_bot_stays_grounded():
    assert_metric(node, faithfulness, threshold=0.9)
```

plain assertions ~ no custom runner, no plugin. works with pytest,
unittest, or anythin' that understands `assert`, and every pytest flag
keeps workin' because nothin' got wrapped.

failures carry the diagnosis the metric already computed:

```
Faithfulness scored 0.500, below the threshold of 0.900.
  Unsupported claims:
    - It was built in 1750.
```

## from the command line

```bash
ragrank eval ragrank.yaml
ragrank eval config.json --output result.json
ragrank compare before.json after.json
```

```yaml
dataset: data.csv
metrics:
  - faithfulness
  - name: token_f1
    threshold: 0.9
run:
  max_workers: 8
  cache: true
```

exit 0 passed, 1 a threshold failed, 2 your config was wrong. yaml needs
`pip install pyyaml`; a `.json` config works with nothin' extra.

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
