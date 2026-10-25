# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ragrank is a Python evaluation library for RAG (Retrieval-Augmented Generation) models. It provides metrics, an evaluation pipeline, a CLI and a pytest plugin to assess RAG system output quality. Uses uv for dependency management, source lives in `src/ragrank/` (plus the pytest plugin in `src/ragrank_pytest.py`).

Plans live in `PLAN.md` (the chain) and `plans/NN-*.md`.

## Commands

```bash
# Install
uv sync                                    # core deps
uv sync --group dev                        # dev deps (pytest, ruff, etc.)
uv sync --group docs                       # docs deps (sphinx)
uv sync --extra all                        # every optional integration

# Test
make test-offline                           # tests without OpenAI (used in CI)
make test                                   # all tests (requires OPENAI_API_KEY)
make test-openai                            # only OpenAI-dependent tests
make test-core                              # unit tests only (tests/unit_tests/)
make test-integration                       # integration tests (tests/integrations_tests/)
make code_coverage                          # pytest --cov=ragrank over tests/
uv run pytest tests/unit_tests/test_dataset.py  # single test file
uv run pytest tests/unit_tests/test_dataset.py::test_name -v  # single test

# Lint & Format
make lint                                   # lint src/ only
make lint-test                              # lint src/ and tests/
make format                                 # auto-format with ruff (isort + format)

# Build
make build_dist                             # build distribution packages
make build_docs                             # build both sphinx sites (docs/docs, docs/api_reference)
```

## Dependencies

- Everything is declared in `pyproject.toml` and locked in `uv.lock`; there is no `requirements.txt` anywhere.
- Optional integrations are extras: `openai`, `litellm`, `langchain`, `llama-index`, `hf` (Hugging Face `datasets`), `all`.
- Dependency groups: `dev` (tests, lint) and `docs` (sphinx, themes, and the integrations autodoc imports).
- Read the Docs builds both sites from the `docs` group with `uv sync` (`docs/docs/.readthedocs.yaml`, `docs/api_reference/.readthedocs.yaml`).
- Dependabot keeps `uv` and GitHub Actions current (`.github/dependabot.yml`).

## Code Style

- **Line length**: 69 characters (ruff), 79 max (pycodestyle)
- **Linter**: ruff with extensive rule set (see `[tool.ruff.lint]` in pyproject.toml)
- `S101` (assert usage) is ignored in tests
- All commands run through `uv run --group dev` (aliased as `PYTHON_EXEC` in Makefile)
- **Python 3.10+ modern typing**: Use builtin types for annotations, not `typing` generics. Common mistakes to avoid:
  - `list[str]` not `List[str]`, `dict[str, Any]` not `Dict[str, Any]`, `tuple[int, ...]` not `Tuple[int, ...]`
  - `type[Foo]` not `Type[Foo]`, `X | None` not `Optional[X]`
  - Import `Iterator`, `Sequence` from `collections.abc`, not `typing`
  - Use `from __future__ import annotations` when needed for forward references

## Architecture

### Core Data Flow

1. User creates a `Dataset` (collection of `DataNode`s) via the readers in `dataset/reader.py`: `from_dict()`, `from_records()`, `from_json()`, `from_jsonl()`, `from_csv()`, `from_dataframe()`, `from_hfdataset()` (with `ColumnMap` to rename columns)
2. User calls `evaluate(data, *, llm=None, metrics=None, run_config=None, target=None)` — the main entry point (`evaluation/base.py`). With a `target`, `data` can be a list of questions and the target produces the responses and context
3. The runner (`evaluation/runner.py`) scores each DataNode × metric under `RunConfig` (concurrency, retries, progress); a row the judge can't score records `None` plus an error instead of failing the run
4. Returns `EvalResult` (`evaluation/outputs.py`) with `scores`, full `MetricResult`s in `results`, `summary()`, `passed`, `cost()`, and `to_dict()` / `to_dataframe()` / `to_json()` / `to_html()`

### Key Abstractions

- **`DataNode`** (`dataset/base.py`): Single evaluation point — `question`, `context` (list of strings), `response`, optional `reference`, `retrieved_ids`, `reference_ids`
- **`Dataset`** (`dataset/base.py`): Collection of DataNodes with I/O converters (`to_records()`, `to_json()`, `to_jsonl()`) and `with_progress()` for tqdm
- **`BaseLLM`** (`llm/base.py`): Abstract LLM interface with `LLMConfig` and `LLMResult`. `default_llm()` returns `OpenaiLLM` (requires `OPENAI_API_KEY`). `FakeLLM` (`llm/fake.py`) for tests; `CachedLLM` with `MemoryCache` / `DiskCache` (`llm/cache.py`)
- **`BaseEmbedding`** (`embedding/base.py`): Embedding interface, `cosine_similarity`, and `FakeEmbedding` for tests
- **`BaseMetric`** (`metric/base.py`): Abstract metric with `score(datanode) -> MetricResult`, a configurable `score_range`, `MetricType` (`BINARY` or `NON_BINARY`) and `CostTier`. Subclasses: `LLMMetric`, `ChunkwiseLLMMetric`, `DeterministicMetric`
- **`Prompt`** (`prompt/base.py`): Structured prompt with name, instructions, examples, input/output keys. Pre-defined prompts in `prompt/_prompts.py`
- **Targets** (`target/base.py`): `run_target()` calls the system under test and normalises its output into `TargetOutput`

### Built-in Metrics (in `metric/`)

- `_response_related/`: `response_relevancy`, `response_conciseness`, `correctness`
- `_context_related/`: `context_relevancy`, `context_utilization`
- `_claims/`: `faithfulness`, `context_reliance`
- `_retrieval/`: `hit_rate`, `mrr`, `mean_average_precision`, `precision_at_k`, `recall_at_k`, `ndcg`
- `_heuristic/`: `exact_match`, `string_presence`, `token_f1`, `rouge_l`, `levenshtein_ratio`, `json_valid`, `semantic_similarity`
- `_safety/`: `pii_free`, `safety`, `answered`
- `_custom/`: `CustomMetric`, `CustomInstruct` (with `InstructConfig`), `LLMJudge`, `Guidelines`, `FunctionMetric` / `@metric`, `Jury`, `Pairwise`

### Evaluation extras (in `evaluation/`)

- `compare.py`: `compare(before, after)` -> `Comparison` of two runs
- `alignment.py`: `align()` a judge against human labels (Pearson, Spearman, Cohen's kappa)
- `report.py`: HTML reports
- `usage.py`: token and cost accounting

### Integrations (`integrations/`)

- **OpenAI** (`openai/`): Default LLM, wraps openai client
- **LiteLLM** (`litellm/`): any provider LiteLLM supports
- **LangChain** (`langchain/`): `LangchainLLMWrapper` adapts LangChain LLMs
- **LlamaIndex** (`llama_index/`): `LlamaIndexLLMWrapper` adapts LlamaIndex LLMs

### CLI and testing

- **CLI** (`cli.py`, entry point `ragrank`): `ragrank eval <config.yaml|json> [-o out] [--html report.html]` and `ragrank compare <baseline> <candidate>`
- **Assertions** (`testing/assertions.py`): `assert_metric`, `assert_evaluation`, `assert_no_regression`
- **pytest plugin** (`src/ragrank_pytest.py`): registers the `ragrank` marker and `--ragrank-report`

### Cross-cutting

- **Pydantic bridge** (`bridge/pydantic.py`): Centralizes all Pydantic imports — use this instead of importing pydantic directly
- **Exceptions** (`exceptions.py`): `RagRankError`, `EvaluationError`, `ValidationError`
- **Constants** (`constants.py`): Default field names, LLM model, embedding dimensions

## Environment

- Requires `OPENAI_API_KEY` environment variable for OpenAI integration and the OpenAI tests
- Python >=3.10, <4.0
