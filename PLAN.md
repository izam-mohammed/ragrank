# ragrank: what is left to build, in order

ragrank is already usable and on PyPI; what is built is in
`docs/built/BUILT-2026-10-22-1829.md`. These plans bring the repository up
to the shared standard: the check first, so every later PR runs it, then
the README, then the site.

### Phase 1: one check on every pull request

- [ ] `plans/01-quality-gates.md` - `make check` (format, lint, tests over
  a 95 % coverage floor, dependency audit) locally and in CI, beside the
  shared security gate.

### Phase 2: the front door

- [ ] `plans/02-readme-banner.md` - a one-screen README with a themed SVG
  banner, the detail moved to the docs.
- [ ] `plans/03-site.md` - a one-page site in `site/`, published on GitHub
  Pages.
