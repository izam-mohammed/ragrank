# every pull request runs one check: lint, format, tests over a 95 % coverage floor, and a dependency audit, beside the shared security gate

## The ask

"Proper lint/format, tests, test coverage floor 90 % (target 95 %),
security checks (dependency audit, secret scan, static analysis where the
language has one), all run by one local check command and by CI on every
PR", reusing the shared gate in `izam-mohammed/.github` rather than
inventing a parallel one.

Built so far: `docs/built/BUILT-2026-10-22-1829.md`.

What the tree shows today (`docs/standards.md`, measured in a scratch copy
on 22/10/2026):

- `make lint` runs ruff 0.2.2 on `src/` and passes; `ruff check tests`
  reports 177 findings (146 are `D103`, a missing test docstring).
- `ruff format --check src tests` passes, but nothing enforces it.
- `make test-offline` passes: 862 passed, 1 skipped, about 23 s.
  Line coverage over `src/ragrank` is 97 %, with no floor.
- No dependency audit, no secret scan. `pip-audit` over `uv export
  --all-extras --group dev` finds 39 advisories in 7 packages (aiohttp,
  banks, certifi, multidict, nltk, pillow, urllib3); the core install has
  none. GitHub's dependency updates bump `docs/requirements.txt` only, so
  `uv.lock` never moves.
- CI (`lint.yml`, `tests.yml`) runs on `push`, never on `pull_request`,
  Python 3.11 only, with actions on moving tags.

## Alternatives

### Decision 1: what audits dependencies, and does it block? (two-way door)

The gate's stage 2 (osv-scanner) reports to the PR without failing it, and
does not run locally. The ask wants the audit in the one local check too.

Searched 2026-10-22: "osv-scanner uv.lock support vs pip-audit uv
export", "pip-audit uv.lock support pylock uv export requirements"; the
uv export docs (docs.astral.sh/uv/concepts/projects/export), the
OSV-Scanner v2 notes on `uv.lock` extraction, and `pip-audit` run here
against `uv export` output.

| Must | Shared gate only (osv-scanner, reports) | `pip-audit` on `uv export` in `make check`, plus the gate | `osv-scanner` locally in `make check`, plus the gate |
| --- | --- | --- | --- |
| Reads this lockfile | yes (`uv.lock` direct) | yes (measured: export then audit) | yes |
| No published advisory against the tool | yes | yes | yes |
| **Survives** | **yes** | **yes** | **yes** |

| Criterion | Weight | Shared gate only | `pip-audit` in `make check` + gate | `osv-scanner` in `make check` + gate |
| --- | ---: | ---: | ---: | ---: |
| Performance efficiency | 40 | 5 | 4 | 4 |
| Fit to the ask | 30 | 2 | 5 | 5 |
| Reliability | 15 | 4 | 4 | 4 |
| Compatibility | 15 | 5 | 4 | 3 |
| **Weighted total** | **100** | **3.95** | **4.30** | **4.15** |

- Gate only, efficiency 5: nothing added to the local run. Fit 2: it never
  blocks and never runs in `make check`.
- `pip-audit` efficiency 4: one `uvx pip-audit` call over an exported
  requirements file; its run time here was not timed, so the 4 is a
  judgement against a no-op.
- `pip-audit` compatibility 4: runs through `uvx`, which the project
  already needs; `osv-scanner` compatibility 3: a separate Go binary every
  contributor must install (it was not installed on this machine).

**Chosen: `pip-audit` on `uv export` in `make check`, blocking, with the
shared gate's osv-scanner still reporting on every PR.** It is the only
option that makes the audit part of the local check without a new binary.
Revisit if the gate's stage 2 is made blocking for Python repos.

### Decision 2: how is the coverage floor held? (two-way door)

Searched 2026-10-22: "pytest-cov fail_under coverage floor vs diff-cover
vs codecov patch coverage"; the diff-cover PyPI page, the coverage.py
`[report] fail_under` setting already used by `[tool.coverage.report]`.

| Criterion | Baseline: reported only | `fail_under` in `[tool.coverage.report]` | diff-cover on changed lines | Codecov patch status |
| --- | --- | --- | --- | --- |
| Performance efficiency | free | free (read at report time) | a second pass over an XML report | an upload per run |
| Fit to the ask (a whole-project floor) | no | yes | no (changed lines only) | no (changed lines only) |
| Running cost and operability | none | none | none | an outside service and a token |

**Chosen: `fail_under = 95` in `[tool.coverage.report]`, run by
`pytest --cov`.** The ask is a whole-project floor; 95 % is the target,
and today's 97 % already clears it with room for one awkward branch. The
90 % minimum in the ask is the line it may never be lowered past. Revisit
if a large optional integration lands that cannot be tested offline.

### Decision 3: what does static analysis? (two-way door)

Searched 2026-10-22: "ruff flake8-bandit S rules vs bandit vs semgrep
python static analysis comparison"; the Semgrep maintainers' Bandit
comparison, the ruff issue on reimplementing Bandit as `S`.

| Criterion | Baseline: ruff `S` rules in the lint, plus the gate's semgrep (reports) | Add bandit to `make check` | Run semgrep in `make check` |
| --- | --- | --- | --- |
| Performance efficiency | ruff ran over `src/` in 0.08 s here | a second Python AST pass | a separate engine and rule download |
| Fit to the ask | security rules per file, plus cross-rule semgrep on the PR | the same rules ruff already runs (one low finding, B105, here) | the same as the gate, run twice |
| Compatibility | already configured | duplicate rule IDs | needs semgrep installed locally |

**Chosen: keep ruff's `S` rules in the lint and the gate's semgrep stage.**
Bandit duplicates `S`; semgrep already runs on every PR through the gate.
Revisit if semgrep's stage is made blocking.

### Decision 4: how does CI run the check? (two-way door)

Searched 2026-10-22: "astral-sh/setup-uv python-version matrix
pull_request workflow pin action commit SHA zizmor"; the pydevtools
guide to pinning actions by SHA, the setup-uv v8 notes (moving `v8` tags
no longer published), and the shared gate's README conventions.

| Criterion | Baseline: `lint.yml` + `tests.yml` on `push` | One `check.yml` running `make check` on `pull_request` and `push` to `main` |
| --- | --- | --- |
| Performance efficiency | two jobs each installing everything | one install per Python version |
| Fit to the ask | never runs on a PR from a fork, no audit, no format | the same command as local, on every PR |
| Security | moving tags, default token permissions | SHAs pinned, `permissions: {}` with `contents: read` per job, as the gate's README asks |

**Chosen: one `check.yml` with a Python 3.10 and 3.13 matrix** (the
lowest supported and the newest), plus `security.yml` calling the shared
gate pinned by SHA. Revisit if a supported version between them breaks
alone.

## The decision

No rough cut in the product sense: nothing a user of the library sees
changes. The rough cut is the gate itself, end to end: `make check` runs
locally and the same command blocks a PR.

- `make check` runs, in order: `ruff format --check`, `ruff check src
  tests`, `pytest -m "not openai" --cov` with the floor (decision 2), the
  dependency audit (decision 1). It stops at the first failure.
- Tests are linted with the docstring and annotation rules relaxed for
  `tests/` (`D1`, `ANN` per-file ignores), the rules the code already
  passes; the rest of the 177 findings are fixed, not ignored.
- `uv.lock` is upgraded for the 7 flagged packages so the audit starts
  green; Dependabot is pointed at `uv` and `github-actions` as in
  `izam-mohammed/repo-template`'s `.github/dependabot.yml`.
- CI per decision 4; secrets and static analysis by the shared gate
  (decision 3), gitleaks blocking.

Dependencies and permissions: `pip-audit` run through `uvx`, not added to
the project; the reusable workflow `izam-mohammed/.github` at a pinned SHA;
CI token `contents: read`, and `pull-requests: write` for the gate's
comment only. No new secret.

Standards: `docs/standards.md` (today `make lint` and `make test-offline`;
this plan adds `make check` and rewrites the file in Phase 1).

## Where it lands

| Path | New or edit | Phase | What changes | Firm? |
| --- | --- | --- | --- | --- |
| `Makefile` | edit | 1 | a `check` target, and `audit` | reuse |
| `pyproject.toml` | edit | 1 | `fail_under`, per-file ignores for `tests/` | reuse |
| `uv.lock` | edit | 1 | the flagged packages upgraded | expected |
| `.github/workflows/check.yml` | new | 1 | `make check` on PRs and `main` | expected |
| `.github/workflows/security.yml` | new | 1 | calls the shared gate, `ecosystem: python` | fixed |
| `.github/dependabot.yml` | new | 2 | `uv` and `github-actions` updates | expected |
| `.github/workflows/lint.yml` | edit | 2 | removed once `check.yml` is required | expected |
| `.github/workflows/tests.yml` | edit | 2 | removed once `check.yml` is required | expected |
| `docs/standards.md` | edit | 1 | the new checks and `make check` | reuse |
| `CLAUDE.md` | edit | 2 | `make check` in Commands, Python `>=3.10` | reuse |

*Reused, not rewritten:*

- `izam-mohammed/.github` `.github/workflows/security-scan.yml`: the shared
  gate (gitleaks blocking; osv-scanner, semgrep and zizmor reporting); the
  calling file copies `izam-mohammed/repo-template`'s `security.yml`.
- `Makefile` `lint`, `lint-test`, `test-offline`, `code_coverage`: `check`
  calls these rather than new commands.
- `[tool.coverage.run]` and `[tool.coverage.report]` in `pyproject.toml`:
  the floor is one key added there.
- Searched for an existing audit or check target (`audit`, `check`,
  `security` in `Makefile`): none.

## Security

What is protected: the PyPI package (published from a release by
`pypi-release.yml` with its existing PyPI token secret, untouched here) and every user who
installs it. Entry points: pull requests, including from forks, and the
dependency updates bots open.

| Threat | Here | Handled by | Proved by |
| --- | --- | --- | --- |
| Tampering: a moved action tag runs new code in CI | `uses:` on `@v4`, `@v5` tags today | every action pinned by full SHA with the version in a comment | the gate's zizmor stage reports no unpinned use on the PR |
| Information disclosure: a secret committed | any PR | gitleaks, blocking, in the shared gate | the gate's stage 1 green on the PR |
| Elevation of privilege: a PR workflow with a write token | `fun.yml` on `pull_request_target`; new workflows | new workflows `permissions: {}` at the top, `contents: read` per job; `fun.yml` left as a follow-up | zizmor's report on the PR |
| Tampering: a vulnerable dependency shipped | `uv.lock` (39 advisories in extras and dev) | the blocking audit in `make check` | `make check` green with the upgraded lock |

No secret is passed to the gate (`secrets: inherit` never used). Logs: CI
prints tool output only. Checklist: SLSA build track and the CIS GitHub
benchmark's workflow items, current editions, for CI.

## Declined

- Shared gate alone for the audit (decision 1): never blocks and never runs
  in `make check`.
- `osv-scanner` in `make check` (decision 1): needs a separate binary on
  every machine; `pip-audit` reaches the same lock through `uvx`.
- diff-cover and Codecov (decision 2): measure changed lines, not the
  whole-project floor asked for; Codecov adds a service and a token.
- bandit and a local semgrep (decision 3): duplicate ruff's `S` rules and
  the gate's semgrep stage.
- Keeping `lint.yml` and `tests.yml` (decision 4): no PR trigger, no audit,
  no format check.

## What the tests hold

The existing 862 offline tests, now under a 95 % floor that fails the run
if coverage drops; the audit failing on any advisory in `uv.lock`; the
format check failing on any unformatted file. Each is proved once in
Phase 1 by making it fail on purpose in a scratch branch.

## Follow-ups

- ruff is pinned at 0.2.2 (February 2024); moving to a current release
  changes rule names (`ANN101` is gone) and is its own change.
- `fun.yml` uses `pull_request_target` with unpinned actions; remove or
  rewrite it.
- The `dev` group still lists `isort`, `flake8`, `bandit` and `codespell`,
  which no command runs.
- No type checker runs; choosing one (mypy or Pyright) is its own plan.

## Repository knowledge

Read root `KNOWLEDGE.md` when present before each phase. Follow the global and
repository instructions for recording useful development findings.

## Phases

### Phase 1: the rough cut: one check, locally and on every PR (needs you: try it)

- [ ] `make check` runs format check, lint over `src` and `tests`, offline
  tests with `fail_under = 95`, and `pip-audit` over `uv export
  --all-extras --group dev` (expected in `Makefile`, `pyproject.toml`),
  with `tests/` given `D1` and `ANN` per-file ignores and the remaining
  test lint findings fixed.
- [ ] `uv.lock` upgraded for aiohttp, banks, certifi, multidict, nltk,
  pillow and urllib3 until the audit reports none, and the offline tests
  still pass (`uv.lock`).
- [ ] `.github/workflows/check.yml` runs `make check` on `pull_request`
  and on `push` to `main`, Python 3.10 and 3.13, actions pinned by SHA,
  `permissions: {}` with `contents: read` on the job; and
  `.github/workflows/security.yml` calls
  `izam-mohammed/.github/.github/workflows/security-scan.yml` pinned by SHA
  with `ecosystem: python`.
- [ ] `docs/standards.md` rewritten for `make check` and each new check.
- [ ] Check: `make check` passes locally; on a throwaway branch, dropping
  `fail_under` to a value above the measured total makes it fail, then it
  is put back.
- [ ] You try it: run `make check` and open the PR; say whether the run
  time and the PR checks (check and security) are what you want.

### Phase 2: tidy the rest of CI

- [ ] `.github/workflows/lint.yml` and `.github/workflows/tests.yml`
  removed once `check.yml` is green on `main`, and `.github/dependabot.yml`
  added for `uv` and `github-actions`, as in `izam-mohammed/repo-template`.
- [ ] `CLAUDE.md` names `make check` and says Python `>=3.10`, matching
  `pyproject.toml`.
- [ ] Check: `make check` passes, and the PR shows the check and security
  workflows only, both green.

### Phase 3: open the PR

- [ ] The branch pushed and its PR opened against the branch it was cut from
  (or, if one is already open for this branch, its body brought up to date)
  with a Conventional Commit title for the whole change, a body naming this
  plan, what changed and why, and the checks run with their result. The PR's
  URL printed as the evidence. Opened, never merged.
