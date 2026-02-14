# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ragrank is a Python evaluation library for RAG (Retrieval-Augmented Generation) models. It provides metrics and an evaluation pipeline to assess RAG system output quality. Uses uv for dependency management, source lives in `src/ragrank/`.

## Commands

```bash
# Install
uv sync                                    # core deps
uv sync --group dev                        # dev deps (pytest, ruff, etc.)
uv sync --group docs                       # docs deps (sphinx)

# Test
make test                                   # all tests
make test-core                              # unit tests only (tests/unit_tests/)
make test-integration                       # integration tests (tests/integration_tests/)
uv run pytest tests/unit_tests/test_dataset.py  # single test file
uv run pytest tests/unit_tests/test_dataset.py::test_name -v  # single test

# Lint & Format
make lint                                   # lint src/ only
make lint-test                              # lint src/ and tests/
make format                                 # auto-format with ruff (isort + format)

# Build
make build_dist                             # build distribution packages
make build_docs                             # build sphinx documentation
```

## Code Style

- **Line length**: 69 characters (ruff), 79 max (pycodestyle)
- **Linter**: ruff with extensive rule set (see `[tool.ruff.lint]` in pyproject.toml)
- `S101` (assert usage) is ignored in tests
- All commands run through `uv run` (aliased as `PYTHON_EXEC` in Makefile)

## Architecture

### Core Data Flow

1. User creates a `Dataset` (collection of `DataNode`s) via `from_dict()`, `from_csv()`, `from_dataframe()`, or `from_hfdataset()`
2. User calls `evaluate(dataset, llm=None, metrics=None)` — the main entry point
3. For each DataNode × metric: the metric formats a `Prompt`, sends it to the `BaseLLM`, parses the score
4. Returns `EvalResult` with scores, convertible to DataFrame/dict

### Key Abstractions

- **`DataNode`** (`dataset/base.py`): Single evaluation point — `question`, `context` (list of strings), `response`
- **`Dataset`** (`dataset/base.py`): Collection of DataNodes with I/O converters and `with_progress()` for tqdm
- **`BaseLLM`** (`llm/base.py`): Abstract LLM interface. `default_llm()` returns `OpenaiLLM` (requires `OPENAI_API_KEY` env var)
- **`BaseMetric`** (`metric/base.py`): Abstract metric with `score(datanode) -> MetricResult`. Types: `BINARY` or `NON_BINARY`
- **`Prompt`** (`prompt/base.py`): Structured prompt with name, instructions, examples, input/output keys. Pre-defined prompts in `prompt/_prompts.py`
- **`EvalResult`** (`evaluation/outputs.py`): Aggregated scores with `to_dataframe()` and `to_dict()`

### Built-in Metrics (in `metric/`)

- `response_relevancy`, `response_conciseness` — in `_response_related/`
- `context_relevancy`, `context_utilization` — in `_context_related/`
- `CustomMetric`, `CustomInstruct` (with `InstructConfig`) — in `_custom/`

### Integrations (`integrations/`)

- **OpenAI** (`openai/`): Default LLM, wraps openai client
- **LangChain** (`langchain/`): `LangchainLLMWrapper` adapts LangChain LLMs
- **LlamaIndex** (`llama_index/`): `LlamaIndexLLMWrapper` adapts LlamaIndex LLMs

### Cross-cutting

- **Pydantic bridge** (`bridge/pydantic.py`): Centralizes all Pydantic imports — use this instead of importing pydantic directly
- **Exceptions** (`exceptions.py`): `RagRankError`, `EvaluationError`, `ValidationError`
- **Constants** (`constants.py`): Default field names, LLM model, embedding dimensions

## Environment

- Requires `OPENAI_API_KEY` environment variable for OpenAI integration and most tests
- Python >=3.9, <4.0
