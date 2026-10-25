# ragrank: what every change passes

Recorded 22/10/2026 from what the project already runs; nothing here was
changed to write it. There is no single check command yet: today a change
is done when `make lint` and `make test-offline` pass, which is what CI
runs. `plans/01-quality-gates.md` adds `make check`, a coverage floor and
the security checks, and updates this file when it lands.

## Toolchain

| What | Version | Pinned in |
| --- | --- | --- |
| Python | `>=3.10,<4.0`; CI uses 3.11 | `pyproject.toml` (`requires-python`), `.github/workflows/*.yml` |
| uv | unpinned (`astral-sh/setup-uv@v5` in CI) | `uv.lock`, committed |
| Build backend | hatchling with hatch-vcs, version from the git tag | `pyproject.toml` (`[build-system]`, `[tool.hatch.version]`) |

## Checks

| Check | Tool | Config | Command | Passing means |
| --- | --- | --- | --- | --- |
| Format | ruff 0.2.2 (`ruff format`) | `pyproject.toml` (`[tool.ruff.format]`) | `make format` rewrites; no check mode in CI | not enforced |
| Lint | ruff 0.2.2 | `pyproject.toml` (`[tool.ruff.lint]`, line length 69) | `make lint` (`src/` only); `make lint-test` adds `tests/` | `make lint` exits 0 |
| Types | none | | | not checked |
| Unit and offline tests | pytest | `pytest.ini` (marker `openai`) | `make test-offline` (`pytest -m "not openai" tests`) | all pass |
| Tests needing a key | pytest | `pytest.ini` | `make test-openai`, needs `OPENAI_API_KEY` | all pass; not run in CI |
| Coverage | pytest-cov | `pyproject.toml` (`[tool.coverage.*]`) | `make code_coverage` | reported only, no floor |
| Dependency audit | none | | | not checked |
| Secret scan | none | | | not checked |
| Static analysis | ruff's `S` (flake8-bandit) rules, in the lint | `pyproject.toml` | `make lint` | no findings in `src/` |
| Build | uv with hatchling | `pyproject.toml` | `make build_dist` | builds a wheel and sdist |
| Docs | Sphinx | `docs/docs/conf.py`, `docs/api_reference/conf.py` | `make build_docs` | builds |

## Tests

Tests live in `tests/unit_tests/` (one `test_<area>.py` per area, all
offline, using `FakeLLM`) and `tests/integrations_tests/` (LangChain,
LlamaIndex, OpenAI wrappers). Tests that call OpenAI carry the `openai`
marker. One test: `uv run pytest tests/unit_tests/test_dataset.py::<name>
-v`. `tests/measurements/benchmark_eval.py` is a benchmark, not collected
as a test.

## Constraints

- A library on PyPI that others depend on: the public API is documented on
  Read the Docs and changes are announced in the GitHub release notes.
- Optional integrations are extras; the core install is `pydantic`,
  `typing-extensions`, `pandas` and `tqdm` only.
- Tests in CI must run without any API key (`make test-offline`).

## Conventions

- Commits: Conventional Commits (`feat:`, `fix:`, `docs:`, `chore(deps):`,
  `ci:`), lowercase, merged through pull requests with the PR number in the
  subject.
- Code style in `CLAUDE.md`: builtin generics, Pydantic only through
  `src/ragrank/bridge/pydantic.py`.
- PRs use `.github/PULL_REQUEST_TEMPLATE/pull_request_template.md`.

## CI

- `.github/workflows/lint.yml`: `make lint` on every `push`.
- `.github/workflows/tests.yml`: `make test-offline` on every `push`.
- `.github/workflows/pypi-release.yml`: `uv build` and `uv publish` when a
  GitHub release is created (uses the `PYPI_TOKEN` secret).
- `.github/workflows/build-docs.yml`: calls the Read the Docs webhooks when
  a release is created.
- `.github/workflows/fun.yml`: a first-interaction greeting bot.

## Decided

- ruff for lint and format, over flake8, isort and black (all still listed
  in the `dev` group): already configured and passing on `src/`.
- uv with `uv.lock` for dependencies: PR `#64`.
- Sphinx with Furo on Read the Docs for documentation: `docs/`.
