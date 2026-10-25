# the docs build from pyproject.toml and uv.lock, with no requirements file

## The ask

The docs should build from `pyproject.toml` alone: `docs/requirements.txt`
goes, and the docs dependencies in `pyproject.toml` are brought up to date.
The plan and its implementation land together in one PR.

What it means once looked at, 25/10/2026:

- The docs dependencies already live in `pyproject.toml` as the `docs`
  dependency group, locked in `uv.lock`, and `make build_docs` builds with
  `uv run --group docs`.
- `docs/requirements.txt` exists only because both Read the Docs configs
  (`docs/docs/.readthedocs.yaml`, `docs/api_reference/.readthedocs.yaml`)
  install from it. `make build_docs` regenerates it with `uv export` on
  every run, so it churns, and Dependabot bumps it too (#73, #74, #77).
- Three packages in the `docs` group are not used by either `conf.py`:
  `sphinx-autodocgen`, `sphinx-autopackagesummary` and
  `sphinxcontrib-napoleon` (the API reference uses Sphinx's built-in
  `sphinx.ext.napoleon`). `linkify-it-py` has a newer release (2.2.0)
  than the lock holds (2.1.1).

No built snapshot: this changes only how the docs are built, not the
`ragrank` package.

## Alternatives

### Decision 1: how Read the Docs installs the docs dependencies (two-way door)

Searched 25/10/2026: "Read the Docs uv build customization"; read Read the
Docs' build customization page (the `method: uv` install with
`command: sync` and `groups`).

| Must | Baseline: exported `docs/requirements.txt` | `method: uv`, `command: sync`, `groups: [docs]` | Custom `build.jobs` installing uv by hand |
| --- | --- | --- | --- |
| One source of truth with local builds (`uv.lock`) | no (a second, generated copy) | yes | yes |
| Documented by Read the Docs | yes | yes | yes |
| **Survives** | no | **yes** | **yes** |

| Criterion | Weight | `method: uv` | Custom `build.jobs` |
| --- | ---: | ---: | ---: |
| Performance efficiency | 40 | 4 | 4 |
| Reliability | 30 | 5 | 3 |
| Maintainability of the result | 30 | 5 | 2 |
| **Weighted total** | **100** | **4.60** | **3.10** |

- Efficiency 4 each: both install from the lock with uv.
- Reliability: the built-in method is maintained by Read the Docs; a
  hand-written job pins its own uv install and breaks when the image
  changes (3).
- Maintainability: three lines of config against a shell script per
  config (2).

**Chosen: `method: uv` with `command: sync` and `groups: [docs]`.**
Revisit if Read the Docs drops the method.

### Decision 2: where the docs dependencies live (two-way door)

Searched 25/10/2026: the decision 1 search; uv's dependency groups docs.

| Criterion | Weight | Baseline: `docs` group in the root `pyproject.toml` | A separate `docs/pyproject.toml` project |
| --- | ---: | ---: | ---: |
| Performance efficiency | 40 | 5 | 4 |
| Compatibility | 30 | 5 | 2 |
| Maintainability of the result | 30 | 5 | 3 |
| **Weighted total** | **100** | **5.00** | **3.10** |

- Autodoc imports `ragrank`, so the docs need the package and its
  optional integrations installed; the root project already provides
  them. A separate project would need a path dependency on the root and
  its own lock (compatibility 2, efficiency 4: two resolutions).

**Chosen: keep the `docs` group in the root `pyproject.toml`.** Revisit if
the docs ever build without importing the package.

### Decision 3: the Read the Docs image and Python (two-way door)

Searched 25/10/2026: Read the Docs' build customization page (its uv
example uses `ubuntu-26.04` and Python 3.14).

| Criterion | Weight | Baseline: ubuntu-22.04, 3.11 | ubuntu-24.04, 3.13 | ubuntu-26.04, 3.14 |
| --- | ---: | ---: | ---: | ---: |
| Reliability | 50 | 3 | 5 | 3 |
| Compatibility | 50 | 3 | 5 | 3 |
| **Weighted total** | **100** | **3.00** | **5.00** | **3.00** |

- 3.13 is what the local `.venv` builds with today, with 0 warnings on
  both builds (the baseline below), so it is proven. 3.14 is untried here
  for `llama-index` and friends; 3.11 differs from local and resolves an
  older Sphinx.

**Chosen: ubuntu-24.04 with Python 3.13.** Revisit when the local
environment moves to 3.14.

## The decision

Both Read the Docs configs install with `method: uv`, `command: sync`,
`groups: [docs]` (decision 1) on ubuntu-24.04 and Python 3.13 (decision 3),
from the `docs` group in the root `pyproject.toml` (decision 2).
`docs/requirements.txt` and the `uv export` line in `make build_docs` are
removed. The three unused packages leave the `docs` group, and the docs
packages are upgraded in `uv.lock`.

Baseline, 25/10/2026 on Python 3.13.16: `sphinx-build -q` prints 0
warnings for `docs/docs` and 0 for `docs/api_reference`.

Dependencies and permissions: none new; three removed.

Standards: no `docs/standards.md` yet; the checks are the CI workflows
(`lint.yml`, `tests.yml`) and `make build_docs`.

Security: no new input, data, secret or permission; dependencies only
shrink.

## Where it lands

| Path | New or edit | Phase | What changes | Firm? |
| --- | --- | --- | --- | --- |
| `docs/docs/.readthedocs.yaml` | edit | 1 | uv install, new image | fixed |
| `docs/api_reference/.readthedocs.yaml` | edit | 1 | uv install, new image | fixed |
| `docs/requirements.txt` | delete | 1 | gone | fixed |
| `Makefile` | edit | 1 | no `uv export` in `build_docs` | reuse |
| `pyproject.toml` | edit | 1 | three unused docs packages dropped | reuse |
| `uv.lock` | edit | 1 | docs packages upgraded | reuse |
| `plans/04-repo-hygiene.md` | edit | 1 | note that the requirements file is gone | expected |

*Reused, not rewritten:* the `docs` group in `pyproject.toml` and
`make build_docs`; nothing new is added.

## Declined

- Keeping the exported `docs/requirements.txt` (decision 1): a second,
  generated copy of the lock.
- A hand-written uv install in `build.jobs` (decision 1): lost on
  reliability and maintainability.
- A separate `docs/pyproject.toml` (decision 2): autodoc needs the root
  package anyway.
- ubuntu-26.04 with Python 3.14 (decision 3): untried with this
  dependency set.

## What the tests hold

- `make build_docs` builds both sites with the same 0 warnings as the
  baseline, and no longer writes `docs/requirements.txt`.
- `uv lock --check` passes.
- `git grep requirements.txt` finds nothing outside plans.
- After merge, the first Read the Docs build of each site installs with
  uv and passes; this can only be watched once it is on `main`, so it is
  checked on the Read the Docs dashboard after merge, not in a phase.

## Phases

### Phase 1: the docs build from uv only

- [x] Both `.readthedocs.yaml` files install with `method: uv`,
  `command: sync`, `groups: [docs]` on ubuntu-24.04 and Python 3.13.
- [x] `docs/requirements.txt` is deleted and `make build_docs` no longer
  exports it (`Makefile`).
- [x] `sphinx-autodocgen`, `sphinx-autopackagesummary` and
  `sphinxcontrib-napoleon` leave the `docs` group in `pyproject.toml`, and
  the docs packages are upgraded in `uv.lock`.
- [x] `plans/04-repo-hygiene.md` carries a note that
  `docs/requirements.txt` is gone.
- [x] Check: `uv lock --check` passes, `make build_docs` builds both sites
  with 0 warnings and leaves `git status` clean, and `git grep
  requirements.txt` finds nothing outside `plans/`.

### Phase 2: open the PR

- [x] The branch pushed and its PR opened against the branch it was cut
  from (or, if one is already open for this branch, its body brought up to
  date) with a Conventional Commit title for the whole change, a body
  naming this plan, what changed and why, and the checks run with their
  result. The PR's URL printed as the evidence. Opened, never merged.

