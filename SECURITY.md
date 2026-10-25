# Security policy

## Supported versions

ragrank is young, so only the latest release on [PyPI](https://pypi.org/project/ragrank/) gets security
fixes. Please upgrade (`pip install --upgrade ragrank`) before reporting.

| Version | Supported |
|---|---|
| latest release | yes |
| older | no |

## Reporting a vulnerability

Please do not open a public issue, pull request or discussion for a security problem.

Report it privately through GitHub:
[Report a vulnerability](https://github.com/izam-mohammed/ragrank/security/advisories/new)
(the "Security" tab, then "Report a vulnerability").

Helpful details:

- the version and how you installed it, with any extras (`openai`, `litellm`, `langchain`, `llama-index`, `hf`)
- what an attacker could do, and what they need first (for example a crafted dataset file, a malicious model
  response or a poisoned cache)
- the smallest steps or script that show it
- any fix you have in mind

Please leave out real API keys and real customer data.

## What happens next

This is a project with one maintainer, so there are no guaranteed response times. The aim is to:

- acknowledge the report
- agree on whether it is a vulnerability and how serious it is
- prepare a fix in a private fork, release it, and publish a GitHub security advisory that credits you (unless
  you would rather not be named)

Please give a reasonable chance to release a fix before you share details publicly.

## Scope

In scope: the `ragrank` package, its `ragrank` CLI, its pytest plugin (`ragrank_pytest`), and this repository's
release and docs workflows.
Problems in the LLM providers and frameworks it talks to (OpenAI, LiteLLM, LangChain, LlamaIndex, Hugging Face
datasets) should go to those projects; tell us too if ragrank makes them worse.
There is no bug bounty.
