# CLAUDE.md

Ragrank is a Python evaluation library for RAG models. Source in `src/ragrank/`.

## Commands

- `uv sync --group dev` — install
- `make test-offline` — run tests (no OpenAI)
- `make lint` / `make format` — lint and format with ruff

## Notes

- Python 3.9+; use builtin generics (`list[str]`, `dict[str, Any]`)
- Line length: 69
- `OPENAI_API_KEY` required for OpenAI integration
