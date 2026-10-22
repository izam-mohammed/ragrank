# the README is one screen: a themed banner from this repository, what ragrank does, install, one example, and links to the docs

## The ask

The README should be short, with a banner near the top (light and dark
SVG, made with the `github-banner` skill) or a UI screenshot.

Built so far: `docs/built/BUILT-2026-10-22-1829.md`.

What the tree shows (`README.md` at `00486d8`):

- The README is about 430 lines: install, quick start, then every feature
  in turn (models, metrics, targets, alignment, custom metrics, juries, CI
  gating, run config, cost, caching, compare, pytest, CLI, data, reports).
  Nearly every section already has a page under `docs/docs/` (for example
  `docs/docs/evaluation/caching.md`, `docs/docs/metrics/custom_metrics/
  jury_and_pairwise.md`, `docs/docs/testing/index.md`).
- The banner is a pair of PNGs loaded from
  `raw.githubusercontent.com/Auto-Playground/Ragrank/...`, a different
  repository from `izam-mohammed/ragrank`; the badges, the clone URL and
  the licence link point there too. The same PNGs sit in this tree at
  `docs/docs/_static/imgs/ragrank_dark.png` and `ragrank_light.png`.
- `README.md` is also the PyPI long description (`readme = "README.md"` in
  `pyproject.toml`), so any image must be an absolute URL that PyPI can
  load.

## Alternatives

### Decision 1: what goes at the top? (two-way door)

Searched 2026-10-22: "GitHub README picture element prefers-color-scheme
light dark SVG banner"; GitHub's docs on `<picture>` with
`prefers-color-scheme`, and the dev.to and CSS-Tricks write-ups on SVG
theming in READMEs; the `github-banner` skill's rules.

| Criterion | Baseline: the PNG pair from `Auto-Playground/Ragrank` | A new light and dark SVG pair (`github-banner` skill) | A screenshot of `result.to_html()` |
| --- | --- | --- | --- |
| Performance efficiency | two raster files fetched from another repository | two small text SVGs, no external resources | one raster, large to stay legible |
| Fit to the ask | a logo, says nothing of what it does; served from another repository | says what it does, themed, from this repository | shows a real output, but a report table is unreadable at banner size |
| Reliability | breaks if the other repository moves or changes | lives in this repository | needs re-taking whenever the report changes |
| Compatibility | works on GitHub and PyPI | works on GitHub; on PyPI via absolute raw URLs (the `<picture>` falls back to the light `<img>`) | works on both |

**Chosen: a new light and dark SVG pair made with the `github-banner`
skill**, stored at `assets/banner-light.svg` and `assets/banner-dark.svg`
and referenced by absolute `raw.githubusercontent.com/izam-mohammed/
ragrank/main/` URLs. ragrank has no screen of its own, and the banner can
say what it does. Revisit if the HTML report becomes the thing people
come for.

### Decision 2: where does the detail go? (two-way door)

Searched 2026-10-22: "minimal README python library what to include vs
full docs on Read the Docs pypi long_description"; the Python guide's
documentation chapter, the Real Python lesson on documenting a package.

| Criterion | Baseline: everything in the README | A one-screen README linking the existing Read the Docs pages |
| --- | --- | --- |
| Performance efficiency | about 430 lines to scroll on GitHub and PyPI | one screen; detail loaded only when wanted |
| Fit to the ask | not minimal | minimal, and nothing lost: each section already has a docs page |
| Maintainability of the result | two copies of every example (README and `docs/docs/`) drift | one copy, in the docs |

**Chosen: a one-screen README whose links go to the existing pages on
`ragrank.readthedocs.io`.** Any README section with no docs page gets one
in `docs/docs/` before it leaves the README. Revisit if PyPI readers ask
for the examples back.

## The decision

Rough cut first: the new README with the banner, which you read on
GitHub. Then the move of every section into the docs is checked, and the
links tidied.

- The README keeps the house voice (lowercase, the `~` asides) and holds:
  the banner, one sentence of what it does, `pip install ragrank` and the
  extras line, the quick start, "bring your own model" in three lines, a
  short list of what else it does with each item linking its docs page,
  development (`make check` once `plans/01-quality-gates.md` lands, else
  `make test-offline`), licence.
- Every repository link moves from `Auto-Playground/Ragrank` to
  `izam-mohammed/ragrank` (decision 1).
- The banner concept is new and logged in the skill's `concepts-used.md`,
  as its rules require; none of the taken concepts is reused.

Dependencies and permissions: none. The skill's `scripts/build.py` runs
in a scratch folder while making the banner; nothing is added to the
project.

Standards: `docs/standards.md` (`make lint`, `make test-offline`; `make
check` once `plans/01-quality-gates.md` lands).

Security: no new input, data, secret, dependency or permission; the SVGs
carry no script, link or external resource (the skill's linter refuses
them).

## Where it lands

| Path | New or edit | Phase | What changes | Firm? |
| --- | --- | --- | --- | --- |
| `assets/banner-light.svg` | new | 1 | light banner | fixed |
| `assets/banner-dark.svg` | new | 1 | dark banner | fixed |
| `README.md` | edit | 1 | one screen, links to the docs | reuse |
| `docs/docs/` | edit | 2 | a page for any README section the docs lack | reuse |

*Reused, not rewritten:*

- `docs/docs/**` on Read the Docs: the existing pages take the README's
  detail; the README links them instead of repeating them.
- `/home/izam/.claude-fabulate/skills/github-banner/`: the template,
  palette, build and preview scripts make the SVGs.
- Searched the tree for an existing SVG banner (`*.svg` outside
  `docs/_build`): none, so `assets/` is new.

## Declined

- Keeping the PNG pair from `Auto-Playground/Ragrank` (decision 1): served
  from another repository, and a logo that does not say what ragrank does.
- A screenshot of the HTML report (decision 1): unreadable at banner size
  and goes stale with every report change.
- Keeping every section in the README (decision 2): not minimal, and every
  example exists twice.

## What the tests hold

No code changes. The skill's `build.py` lints both SVGs (no overlap, no
animation, no external resource, contrast). Every README link is
requested and answers 200, checked by a `curl` loop in the phase check.

## Repository knowledge

Read root `KNOWLEDGE.md` when present before each phase. Follow the global and
repository instructions for recording useful development findings.

## Phases

### Phase 1: the rough cut: the one-screen README with its banner (needs you: try it)

- [ ] Three banner concepts drafted per the `github-banner` skill, one
  chosen, built to `assets/banner-light.svg` and `assets/banner-dark.svg`
  with `scripts/build.py` exiting 0, and the concept appended to the
  skill's `concepts-used.md`.
- [ ] `README.md` cut to one screen as `## The decision` lists, banner in a
  `<picture>` with absolute `izam-mohammed/ragrank` raw URLs, every
  `Auto-Playground` link replaced.
- [ ] Check: `make lint` and `make test-offline` pass (or `make check` once
  `plans/01-quality-gates.md` has landed), and every URL in `README.md`
  answers 200 to `curl -sIL`.
- [ ] You try it: open the branch's README on GitHub in light and dark
  theme, and say whether the banner and the length feel right.

### Phase 2: nothing lost from the old README

- [ ] Each section removed from the README is matched to its page in
  `docs/docs/`; a section with no page gets one (expected under
  `docs/docs/evaluation/` or `docs/docs/metrics/`), linked from the README.
- [ ] Check: the project check passes, `make build_docs` builds without a
  new warning, and `uv build` then `uvx twine check dist/*` passes so the
  PyPI description renders.

### Phase 3: open the PR

- [ ] The branch pushed and its PR opened against the branch it was cut from
  (or, if one is already open for this branch, its body brought up to date)
  with a Conventional Commit title for the whole change, a body naming this
  plan, what changed and why, and the checks run with their result. The PR's
  URL printed as the evidence. Opened, never merged.
