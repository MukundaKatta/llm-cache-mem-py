"""llm-cache-mem-py — in-process LRU cache for LLM responses."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable


def _hash_key(messages: list[dict], model: str | None = None,
              extras: dict | None = None) -> str:
    """Produce a stable cache key from messages + model + extras."""
    payload = {"messages": messages, "model": model or "", "extras": extras or {}}
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass
class CacheEntry:
    """A single cached LLM response."""

    key: str
    response: Any
    created_at: float = field(default_factory=time.time)
    ttl: float | None = None   # seconds; None = no expiry
    hit_count: int = 0

    @property
    def expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl


class LLMCache:
    """
    In-process LRU cache for LLM responses.

    Keyed by (messages, model, extras). Thread-safe.

    Example::

        cache = LLMCache(max_size=100, default_ttl=300.0)

        def call_llm(messages):
            key = cache.make_key(messages, model="claude-sonnet")
            if cache.has(key):
                return cache.get(key)
            response = client.chat(messages)
            cache.put(key, response)
            return response

        # Or use the wrap helper:
        @cache.wrap(model="claude-sonnet")
        def call_llm(messages):
            return client.chat(messages)
    """

    def __init__(self, max_size: int = 256, default_ttl: float | None = None) -> None:
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def make_key(
        self,
        messages: list[dict],
        model: str | None = None,
        extras: dict | None = None,
    ) -> str:
        """Compute a cache key for the given messages + params."""
        return _hash_key(messages, model, extras)

    def has(self, key: str) -> bool:
        """Return True if the key exists and is not expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            if entry.expired:
                del self._store[key]
                return False
            return True

    def get(self, key: str) -> Any:
        """Return the cached response, or None if not found/expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.expired:
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            entry.hit_count += 1
            self._hits += 1
            return entry.response

    def put(self, key: str, response: Any, ttl: float | None = None) -> None:
        """Store a response. Evicts the least recently used entry if full."""
        effective_ttl = ttl if ttl is not None else self.default_ttl
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = CacheEntry(key=key, response=response, ttl=effective_ttl)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)  # evict LRU

    def delete(self, key: str) -> bool:
        """Delete a cache entry. Returns True if it existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def prune_expired(self) -> int:
        """Remove expired entries. Returns the number removed."""
        with self._lock:
            expired_keys = [k for k, v in self._store.items() if v.expired]
            for k in expired_keys:
                del self._store[k]
            return len(expired_keys)

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float | None:
        total = self._hits + self._misses
        return self._hits / total if total else None

    @property
    def stats(self) -> dict:
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }

    def wrap(self, model: str | None = None, extras: dict | None = None, ttl: float | None = None):
        """
        Decorator that caches the return value keyed on the first argument (messages).

        The decorated function must take messages as its first positional arg.
        """
        import functools

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(messages, *args, **kwargs):
                key = self.make_key(messages, model=model, extras=extras)
                cached = self.get(key)
                if cached is not None:
                    return cached
                result = fn(messages, *args, **kwargs)
                self.put(key, result, ttl=ttl)
                return result
            return wrapper
        return decorator


__all__ = ["LLMCache", "CacheEntry"]
