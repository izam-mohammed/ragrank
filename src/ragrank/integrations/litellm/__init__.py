"""LiteLLM integration -- one judge, every provider.

Examples::

    from ragrank.integrations.litellm import LiteLLM

    judge = LiteLLM(model="gemini/gemini-2.0-flash")
"""

from ragrank.integrations.litellm.litellm_llm import LiteLLM

__all__ = ["LiteLLM"]
