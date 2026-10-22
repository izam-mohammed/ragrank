# ragrank has a one-page site at izam-mohammed.github.io/ragrank that says what it does and gets a person to `pip install ragrank` and the docs

## The ask

"A `site/` folder holding a presentable one-page site for users", with
`mac-notch/site/index.html` as the reference.

Built so far: `docs/built/BUILT-2026-10-22-1829.md`.

What exists:

- No `site/` folder. The repository's homepage on GitHub is
  `https://ragrank.readthedocs.io/`, the Sphinx docs (`docs/docs/`), which
  open on a documentation index rather than a pitch.
- The reference, `mac-notch/site/index.html` (commit `9e588fc`, 140 lines,
  about 7.8 KB): one self-contained file, inline CSS, no scripts, no web
  fonts, an inline SVG favicon, a header with two links, one hero with a
  single call to action (the latest release download), and a footer. How
  it is published is not recorded in that repository (it is private and
  has no Pages site), so publishing is decided here.
- GitHub Pages is not enabled on `izam-mohammed/ragrank`
  (`gh api repos/izam-mohammed/ragrank/pages` answers 404). The owner's
  user site, `izam-mohammed.github.io`, publishes a folder with
  `actions/deploy-pages` from `.github/workflows/static.yml`.

## Alternatives

### Decision 1: where is the site published? (two-way door)

Searched 2026-10-22: "GitHub Pages deploy static folder
actions/deploy-pages vs Cloudflare Pages vs Read the Docs landing page";
the GitHub Pages, Cloudflare Pages and Read the Docs comparisons
(stackshare, Read the Docs' own comparison page), and the owner's
`izam-mohammed.github.io` workflow as a working example.

| Must | GitHub Pages from `site/` by Actions | Cloudflare Pages | The Read the Docs landing (a Sphinx index page) |
| --- | --- | --- | --- |
| No new account or secret | yes (the repository's own token) | no (a Cloudflare account and API token) | yes |
| Serves one hand-written HTML file as is | yes | yes | no (rebuilt through Sphinx and its theme) |
| **Survives** | **yes** | no | no |

**Chosen: GitHub Pages, deployed from `site/` by a workflow on `push` to
`main` that changes `site/**`**, at
`https://izam-mohammed.github.io/ragrank/`. It needs no new account or
secret and serves the file exactly as written. Revisit if the site needs
a custom domain with headers GitHub Pages cannot set.

### Decision 2: what is the page made of? (two-way door)

Searched 2026-10-22: "python library landing page single static html vs
mkdocs material home page vs sphinx index"; pydevtools' guide to Sphinx
and MkDocs, the MkDocs vs Sphinx comparison for 2026, the mac-notch page.

| Criterion | One self-contained `site/index.html`, as mac-notch | A static site generator (MkDocs Material) |
| --- | --- | --- |
| Performance efficiency | one request, about 8 KB for the reference, no script | a theme's CSS and JavaScript bundle per page |
| Fit to the ask | one page, pitch first | built for many-page docs, which Read the Docs already serves |
| Maintainability of the result | one file, no build step | a generator config and a build in CI |
| Security | no script to trust | a theme's script on the page |

**Chosen: one self-contained `site/index.html`**, inline CSS and SVG, no
script, no web font, light and dark from `prefers-color-scheme`. The docs
stay on Read the Docs. Revisit if the site grows past one page.

## The decision

Rough cut first: the page itself, opened from the file, then published.

- `site/index.html`: a header (the name, links to Docs, PyPI, GitHub), a
  hero (one line of what it does in the README's voice, a copyable `pip
  install ragrank`, a primary link to the quick start on Read the Docs),
  one short code sample (the README's quick start), three short
  points taken from the README (free retrieval metrics, judge agreement
  with `align`, gating in CI), and a footer with the licence. No figure
  that the README or a test does not already back.
- The banner from `plans/02-readme-banner.md` is reused as the hero art if
  it has landed; otherwise the page ships with text only.
- `.github/workflows/pages.yml` publishes `site/` with
  `actions/upload-pages-artifact` and `actions/deploy-pages`, pinned by
  SHA, on `push` to `main` touching `site/**` and on `workflow_dispatch`.
- The repository's homepage stays the docs until you say otherwise (see
  `## Open decisions`).

Dependencies and permissions: GitHub Pages switched on for the repository
with "GitHub Actions" as the source (a repository setting, done by you);
the deploy job gets `pages: write` and `id-token: write`. No secret, no
package.

Standards: `docs/standards.md` (`make lint`, `make test-offline`; `make
check` once `plans/01-quality-gates.md` lands). The page itself is checked
locally in each phase: a `grep` confirms no `<script` and no external
resource, and a `curl` loop confirms every link answers.

## Where it lands

| Path | New or edit | Phase | What changes | Firm? |
| --- | --- | --- | --- | --- |
| `site/index.html` | new | 1 | the page | fixed |
| `.github/workflows/pages.yml` | new | 2 | publishes `site/` | expected |
| `README.md` | edit | 2 | links the site | reuse |

*Reused, not rewritten:*

- `mac-notch/site/index.html`: the shape (header, one hero, one call to
  action, footer, inline everything), not its look or words.
- `izam-mohammed.github.io/.github/workflows/static.yml`: the deploy
  steps, with SHAs pinned and permissions per job as
  `izam-mohammed/.github`'s README asks.
- `README.md` quick start and feature text: the page's copy comes from it,
  so no claim is new.
- Searched for an existing site or Pages workflow in this tree (`site/`,
  `pages`, `gh-pages`): none.

## Security

What is protected: the repository's Pages deployment and the visitors to
it. Entry point: a push to `main`; nothing on the page takes input.

| Threat | Here | Handled by | Proved by |
| --- | --- | --- | --- |
| Tampering: a moved action tag deploys something else | `pages.yml` | actions pinned by full SHA | the shared gate's zizmor stage on the PR |
| Elevation of privilege: a PR deploys | `pages.yml` | runs only on `push` to `main` and `workflow_dispatch`, never `pull_request`; `pages: write` on the deploy job only | zizmor's report; the workflow's `on:` block |
| Information disclosure: the page leaks a visitor to a third party | `site/index.html` | no script, no web font, no external image; a `<meta>` Content-Security-Policy of `default-src 'none'; style-src 'unsafe-inline'; img-src data:` | the phase check's `grep` for `<script`, `http` resource URLs and the CSP line |

No personal data, no secret, nothing logged. Checklist: OWASP Top 10
(current edition) for the page, CIS GitHub benchmark items for the
workflow.

## Open decisions

- Should the repository's homepage on GitHub switch from Read the Docs to
  the new site? Default: no, the docs stay the homepage and the README
  links the site.

## Declined

- Cloudflare Pages (decision 1): needs a new account and an API token
  secret.
- A Read the Docs landing page (decision 1): rebuilt through Sphinx and
  its theme, not served as written.
- MkDocs Material (decision 2): a script bundle and a build step for one
  page; Read the Docs already holds the many-page docs.
- An online HTML validator in the check: it sends the file to an outside
  service.

## What the tests hold

No library code changes. The phase checks hold: the page has no script and
no external resource, every link on it answers 200, and the deployed URL
serves the same bytes as `site/index.html`.

## Repository knowledge

Read root `KNOWLEDGE.md` when present before each phase. Follow the global and
repository instructions for recording useful development findings.

## Phases

### Phase 1: the rough cut: the page, opened locally (needs you: try it)

- [ ] `site/index.html` with the header, hero, install line, quick start
  sample, three points and footer as `## The decision` lists, light and
  dark, readable at 360 px wide, with the CSP `<meta>`.
- [ ] Check: the project check passes; `grep -c '<script' site/index.html`
  prints 0; every `href` on the page answers 200 to `curl -sIL`.
- [ ] You try it: open `site/index.html` in a browser, in light and dark
  and at phone width, and say whether the pitch and the look are right.

### Phase 2: published (needs you: switch Pages on)

- [ ] `.github/workflows/pages.yml` deploys `site/` on `push` to `main`
  touching `site/**` and on `workflow_dispatch`, actions pinned by SHA,
  `permissions: {}` at the top and `pages: write`, `id-token: write` on the
  deploy job only.
- [ ] You switch Pages on in the repository's settings with "GitHub
  Actions" as the source; the agent stops and asks here.
- [ ] `README.md` links the site under its banner.
- [ ] Check: the project check passes; after the merge,
  `curl -s https://izam-mohammed.github.io/ragrank/ | cmp - site/index.html`
  reports no difference.

### Phase 3: open the PR

- [ ] The branch pushed and its PR opened against the branch it was cut from
  (or, if one is already open for this branch, its body brought up to date)
  with a Conventional Commit title for the whole change, a body naming this
  plan, what changed and why, and the checks run with their result. The PR's
  URL printed as the evidence. Opened, never merged.
