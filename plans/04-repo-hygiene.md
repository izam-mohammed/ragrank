# the repo keeps its dependencies current and its security policy tells the truth

## The ask

The owner asked, in their own words: "have dependabot in all of my repos",
"have proper security.md file for all of my repos", and "proper licence for
all of the repos ... which is a product in nature ... even for private ones".

In this repository that means: there is no Dependabot config at all, while
the repository has five GitHub Actions workflows and a `uv` project
(`pyproject.toml`, `uv.lock`). `SECURITY.md` exists but is not fine: it
names an address `security@ragrank.com` whose link goes to a personal
mailbox, promises acknowledgement "within 24 hours", and says nothing of
supported versions or scope. The Apache-2.0 `LICENSE`, the `pyproject.toml`
`license = "Apache-2.0"` field and the README's licence line already agree,
so the licence stays as it is. Topics are already accurate.

No built snapshot is written: this change touches only repository metadata
(Dependabot config, a policy file, topics), not the `ragrank` package.

## Alternatives

### Decision 1: which bot keeps dependencies current (two-way door)

The owner named Dependabot. Nothing runs here yet beyond GitHub's security
alerts, so the options are compared from scratch.

Searched 22/10/2026: "Renovate vs Dependabot GitHub app private
repositories minimumReleaseAge", "dependabot pre-commit ecosystem"; read
the Dependabot options reference (docs.github.com), the 10/03/2026 GitHub
changelog on pre-commit support, and Renovate's minimum release age docs
(docs.renovatebot.com).

| Must | Dependabot (user's pick) | Renovate |
| --- | --- | --- |
| Updates uv and Actions | yes | yes |
| Runs without a third-party app on the account | yes | no (needs the Mend app or a self-hosted runner) |
| **Survives** | **yes** | no |

| Criterion | Weight | Dependabot |
| --- | ---: | ---: |
| Performance efficiency | 40 | 4 |
| Reliability | 30 | 4 |
| Compatibility | 30 | 5 |
| **Weighted total** | **100** | **4.30** |

- Dependabot efficiency 4: runs on GitHub's side with no CI minutes; a
  weekly schedule and one group per ecosystem keep it to a few PRs a week.
- Dependabot compatibility 5: `uv` and `github-actions` are documented
  ecosystems, and the owner's other repositories already use it.

Sensitivity: only a need for cross-ecosystem grouping would reopen this.
Risk: a bad release is proposed; the 7-day cooldown limits it.

**Chosen: Dependabot (user's pick).** It needs no outside app and matches
the owner's other repositories. Revisit if grouping across ecosystems
becomes necessary.

## The decision

The topics are left as they are. A new `.github/dependabot.yml` covers
`github-actions` (`ci(deps)`) and `uv` (`chore(deps)`), each weekly with a
7-day cooldown and one group (Decision 1). `SECURITY.md` is rewritten:
only the latest PyPI release is supported, reports go through GitHub
private vulnerability reporting (enabled on this public repository), no
response time is promised, and the scope names the package, its CLI, its
pytest plugin and the release and docs workflows.

New dependencies or permissions: none.

Security: the change adds no code, input or secret; it replaces a policy
that pointed reporters at an address that does not match its link.

## Where it lands

| Path | New or edit | Phase | What changes | Firm? |
| --- | --- | --- | --- | --- |
| `.github/dependabot.yml` | new | 2 | github-actions and uv | fixed |
| `SECURITY.md` | edit | 3 | rewritten policy | fixed |

*Reused, not rewritten:*

- Searched the tree for an existing Dependabot or Renovate config: none.

## Declined

- Renovate: fails the must of running without a third-party app.
- A `pip` entry for `docs/requirements.txt`: the file is generated from
  `uv.lock` by `uv export` (`make` target in `Makefile`), so a separate
  updater would fight the lock file; the `uv` entry covers those packages.
- Rewriting `docs/docs/more/security.md` here: it is the docs site's copy,
  left for a docs change.

## Follow-ups

- `docs/docs/more/security.md` still describes the old email channel; it
  should point to `SECURITY.md` once this lands.

## Phases

### Phase 1: topics

- [x] Topics kept as `evaluation`, `language-model`, `llm`, `llm-eval`,
  `llmops`, `machine-learning`, `prompt-engineering`, `rag`, with `python`
  added; check: `gh repo view --json repositoryTopics` lists them.

### Phase 2: Dependabot

- [x] `.github/dependabot.yml` added with `github-actions` and `uv`
  (weekly, 7-day cooldown, one group each); check: the YAML parses.

### Phase 3: security policy

- [x] `SECURITY.md` rewritten: supported versions, private reporting link,
  what happens next, scope; check: no email address or response time is
  invented in it.

### Phase 4: open the PR

- [x] The branch pushed and its PR opened against `main` (or, if one is
  already open for this branch, its body brought up to date) with a
  Conventional Commit title for the whole change, a body naming this plan,
  what changed and why, and the checks run with their result. The PR's URL
  printed as the evidence. Opened, never merged.
  Landed 22/10/2026: https://github.com/izam-mohammed/ragrank/pull/76
