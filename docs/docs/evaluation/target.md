(system-under-test)=
# 🎯 Running your pipeline

Most of ragrank scores responses that already exist, which quietly
assumes somebody else ran the pipeline and saved the output. That is
fine once. It is the wrong shape for the thing you actually want: hold a
set of questions fixed, change the pipeline, and see what moved.

A **target** is any callable that takes a question and returns what your
RAG system produced for it.

```python
from ragrank import evaluate
from ragrank.metric import faithfulness


def my_rag(question: str) -> tuple[str, list[str]]:
    chunks = retriever.search(question)
    return generator(question, chunks), chunks


result = evaluate(
    ["who wrote the first program?"],
    target=my_rag,
    metrics=[faithfulness],
)
```

The question set is now the fixture and the pipeline is the variable.

## What a target may return

Whichever shape your code already produces:

| Return | Read as |
| --- | --- |
| `"Ada Lovelace"` | an answer, no visible retrieval |
| `("Ada", ["chunk one"])` | `(response, context)` |
| `{"response": ..., "context": [...]}` | by key |
| `{"response": ..., "retrieved_ids": [...]}` | also unlocks the ranking metrics |
| `TargetOutput(response=...)` | as given |

A `context` given as a single string is wrapped into a one-chunk list.

## Re-running an existing dataset

Pass a `Dataset` instead of a list and ragrank reuses its questions and
its references, discarding the stale responses. This is the regression
shape - same questions, new pipeline, comparable numbers:

```python
from ragrank.evaluation import compare
from ragrank.target import run_target

before = evaluate(golden, metrics=[faithfulness])
after = evaluate(run_target(golden, my_new_rag), metrics=[faithfulness])

print(compare(before, after))
```

## Building the dataset without scoring it

`run_target` is available on its own when you want to generate now and
score later:

```python
from ragrank.target import run_target

dataset = run_target(questions, my_rag, references=answers)
dataset.to_jsonl("generated.jsonl")
```

## When the target fails

Generation raises by default, rather than quietly shrinking your
evaluation set. A question with no answer is not partial data the way an
unscored row is - it is a missing row, and a run that silently drops ten
of them reports a number for a dataset you did not evaluate.

```python
from ragrank.target import TargetError

try:
    dataset = run_target(questions, flaky_rag)
except TargetError as error:
    ...
```

For a long generation run that should survive a few bad rows:

```python
dataset = run_target(
    questions,
    flaky_rag,
    max_retries=3,
    skip_failures=True,
)
```

Skipped questions are logged, and if every one fails you still get an
error rather than an empty dataset.

## Concurrency

Targets run concurrently, and the results keep the order of the
questions regardless. `evaluate(..., target=...)` takes its concurrency
and retry policy from the same `RunConfig` the scoring uses.

```python
from ragrank.evaluation import RunConfig

result = evaluate(
    questions,
    target=my_rag,
    run_config=RunConfig(max_workers=8, max_retries=2),
)
```
