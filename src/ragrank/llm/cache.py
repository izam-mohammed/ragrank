"""Caching of language model responses.

Evaluation is unusually cache-friendly: judges run at temperature 0 over
a dataset that barely changes, so re-running after adding one metric
re-asks the other five exactly the same questions. Without a cache you
pay for that every time.

The key is the prompt plus the model identity and the settings that
change its answer, so a cache hit is a request that would genuinely
have produced the same response.
"""

from __future__ import annotations

import json
import logging
from hashlib import sha256
from pathlib import Path
from threading import Lock

from ragrank.bridge.pydantic import ConfigDict, Field, PrivateAttr
from ragrank.llm.base import BaseLLM, LLMConfig, LLMResult

logger = logging.getLogger(__name__)


def cache_key(name: str, config: LLMConfig, text: str) -> str:
    """Build the cache key for one request.

    Args:
        name (str): The model's name.
        config (LLMConfig): The settings used for the call.
        text (str): The prompt.

    Returns:
        str: A hex digest identifying the request.
    """
    payload = json.dumps(
        {
            "model": name,
            "config": config.model_dump(),
            "prompt": text,
        },
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


class CacheBackend:
    """Where cached responses live. Subclass to store them elsewhere."""

    def get(self, key: str) -> str | None:
        """Look up a cached response.

        Args:
            key (str): The cache key.

        Returns:
            str | None: The response, or None on a miss.
        """
        raise NotImplementedError

    def set(self, key: str, value: str) -> None:
        """Store a response.

        Args:
            key (str): The cache key.
            value (str): The response to store.
        """
        raise NotImplementedError


class MemoryCache(CacheBackend):
    """An in-process cache. Lives and dies with the process."""

    def __init__(self) -> None:
        """Start empty."""
        self._lock = Lock()
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        """Look up a cached response.

        Args:
            key (str): The cache key.

        Returns:
            str | None: The response, or None on a miss.
        """
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        """Store a response.

        Args:
            key (str): The cache key.
            value (str): The response to store.
        """
        with self._lock:
            self._store[key] = value

    def __len__(self) -> int:
        """Number of cached responses.

        Returns:
            int: The entry count.
        """
        with self._lock:
            return len(self._store)


class DiskCache(CacheBackend):
    """A cache on disk, so it survives between runs.

    One file per entry, named by key. Deliberately dependency-free and
    dull: a corrupt or unreadable entry is treated as a miss rather than
    an error, because a broken cache must never break a run.

    Attributes:
        directory (Path): Where entries are written.
    """

    def __init__(
        self, directory: str | Path = ".ragrank_cache"
    ) -> None:
        """Open (and create) the cache directory.

        Args:
            directory (str | Path): Where to store entries.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        """Path for one entry."""
        return self.directory / f"{key}.json"

    def get(self, key: str) -> str | None:
        """Look up a cached response.

        Args:
            key (str): The cache key.

        Returns:
            str | None: The response, or None on a miss or a bad entry.
        """
        path = self._path(key)
        try:
            return json.loads(path.read_text(encoding="utf-8"))[
                "response"
            ]
        except (OSError, ValueError, KeyError):
            return None

    def set(self, key: str, value: str) -> None:
        """Store a response.

        A cache that cannot be written is logged and ignored -- losing
        a cache entry is not worth losing a run over.

        Args:
            key (str): The cache key.
            value (str): The response to store.
        """
        try:
            self._path(key).write_text(
                json.dumps({"response": value}), encoding="utf-8"
            )
        except OSError as error:
            logger.warning("could not write cache entry: %s", error)

    def __len__(self) -> int:
        """Number of cached responses.

        Returns:
            int: The entry count.
        """
        return len(list(self.directory.glob("*.json")))


class CachedLLM(BaseLLM):
    """Wraps a language model, serving repeat prompts from a cache.

    Attributes:
        inner (BaseLLM): The model doing the actual work.
        backend (CacheBackend): Where responses are kept.
    """

    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True
    )

    inner: BaseLLM = Field(description="The wrapped language model.")
    backend: CacheBackend = Field(
        repr=False, description="Where responses are kept."
    )

    _hits: int = PrivateAttr(default=0)
    _misses: int = PrivateAttr(default=0)
    _lock: Lock = PrivateAttr(default_factory=Lock)

    @property
    def name(self) -> str:
        """The wrapped model's name.

        Returns:
            str: The name.
        """
        return self.inner.name

    @property
    def hits(self) -> int:
        """How many prompts were served from the cache.

        Returns:
            int: The hit count.
        """
        return self._hits

    @property
    def misses(self) -> int:
        """How many prompts reached the model.

        Returns:
            int: The miss count.
        """
        return self._misses

    def generate_text(self, text: str) -> LLMResult:
        """Return a cached response, or call the model and cache it.

        A cached result reports zero tokens and zero time, because that
        is what it cost.

        Args:
            text (str): The input text.

        Returns:
            LLMResult: The response, cached or fresh.
        """
        key = cache_key(self.inner.name, self.llm_config, text)

        cached = self.backend.get(key)
        if cached is not None:
            with self._lock:
                self._hits += 1
            return LLMResult(
                response=cached,
                response_time=0.0,
                finish_reason="cached",
                prompt_tokens=0,
                response_tokens=0,
                llm=self.inner,
                llm_config=self.llm_config,
            )

        with self._lock:
            self._misses += 1
        result = self.inner.generate_text(text)
        self.backend.set(key, result.response)
        return result
