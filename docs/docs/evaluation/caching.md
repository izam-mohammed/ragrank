(caching)=
# 🗄 Caching

Evaluation is unusually cache-friendly. Judges run at temperature 0
over a dataset that barely changes, so adding one metric re-asks the
other five exactly the same questions and bills you for it.

```python
from ragrank.evaluation import RunConfig

evaluate(data, run_config=RunConfig(cache=True))
```

One flag, and an on-disk cache that survives between runs.

## What it saves

```text
run 1 (cold)       5 calls, 1345 tokens
run 2 (warm)       5 calls, 0 tokens
run 3 (+1 metric)  only the new metric was billed
```

The third line is the everyday case. You change one thing and pay for
one thing.

## Choosing a backend

```python
from ragrank.llm import DiskCache, MemoryCache

RunConfig(cache=True)                      # .ragrank_cache on disk
RunConfig(cache=DiskCache("~/my-cache"))   # somewhere else
RunConfig(cache=MemoryCache())             # this process only
RunConfig(cache=None)                      # off, the default
```

`MemoryCache` is useful in tests and notebooks where you do not want
files appearing. `DiskCache` is the one that pays for itself across
runs.

Caching is **off by default**, so no files appear on disk unless you
ask for them.

## What counts as the same request

The cache key is the prompt, plus the model name, plus the settings
that change the answer. A different temperature is a different request
and will not hit. A hit is a request that would genuinely have returned
the same response.

## A cached result is honest about its cost

```python
cached.prompt_tokens    # 0
cached.response_tokens  # 0
cached.response_time    # 0.0
cached.finish_reason    # "cached"
```

That keeps the [cost reporting](./cost.md) accurate rather than
double-counting cache hits as though you had paid for them again.

## A broken cache never breaks a run

A corrupt entry is treated as a miss. A cache directory that cannot be
written is logged and ignored. Losing a cache entry is not worth losing
an evaluation over.

## Writing your own backend

```python
from ragrank.llm import CacheBackend

class RedisCache(CacheBackend):
    def get(self, key: str) -> str | None:
        ...

    def set(self, key: str, value: str) -> None:
        ...
```
