(integration-third-party)=
# Third-party

Packages built on top of ragrank by other people. They live outside
this repository and are maintained by their authors, not by ragrank.
Read their docs and code before you depend on one.

## EvalPort (openeval) adapter

[`ragrank-openeval-adapter`](https://github.com/adhabnr-ux/evalport/tree/main/adapters/ragrank-openeval-adapter)
converts a ragrank `EvalResult` into an [EvalPort](https://github.com/adhabnr-ux/evalport)
suite and result set, and an EvalPort suite back into a ragrank
`Dataset`. Discussed in
[issue #63](https://github.com/izam-mohammed/ragrank/issues/63).

It is not on PyPI yet; install it from source:

```bash
pip install "ragrank-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/ragrank-openeval-adapter"
```

See the package's README for `to_openeval()`, `from_openeval()` and how
it handles unscored rows and metric score ranges.
