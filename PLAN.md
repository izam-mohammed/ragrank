# ragrank: what is left to build, in order

ragrank is already usable and on PyPI; what is built is in
`docs/built/BUILT-2026-10-22-1829.md`. These plans bring the repository up
to the shared standard: the check first, so every later PR runs it, then
the site. The README and its banner stay as they are: judged good enough
on 22/10/2026, so no README plan.

### Phase 1: one check on every pull request

- [ ] `plans/01-quality-gates.md` - `make check` (format, lint, tests over
  a 95 % coverage floor, dependency audit) locally and in CI, beside the
  shared security gate.

### Phase 2: the front door

- [ ] `plans/03-site.md` - a one-page site in `site/`, published on GitHub
  Pages.

### Phase 3: repo hygiene

- [x] `plans/04-repo-hygiene.md` - Dependabot for every ecosystem, a truthful
  security policy and topics.
