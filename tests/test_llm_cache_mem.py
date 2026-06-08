"""Tests for llm-cache-mem-py.

Uses only the standard-library ``unittest`` framework so the suite runs with no
third-party dependencies::

    python3 -m unittest discover -s tests
"""

import os
import sys
import time
import unittest

# Make the src-layout package importable when the suite is run directly via
# ``python3 -m unittest discover -s tests`` without first installing the package.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llm_cache_mem import CacheEntry, LLMCache  # noqa: E402

MSGS = [{"role": "user", "content": "Hello"}]


class PutGetTests(unittest.TestCase):
    def test_put_and_get(self):
        cache = LLMCache()
        key = cache.make_key(MSGS)
        cache.put(key, {"content": "Hi!"})
        self.assertEqual(cache.get(key), {"content": "Hi!"})

    def test_get_miss(self):
        cache = LLMCache()
        self.assertIsNone(cache.get("nonexistent"))

    def test_put_overwrites_existing_key(self):
        cache = LLMCache()
        key = cache.make_key(MSGS)
        cache.put(key, "first")
        cache.put(key, "second")
        self.assertEqual(cache.get(key), "second")
        self.assertEqual(cache.size, 1)

    def test_has_true(self):
        cache = LLMCache()
        key = cache.make_key(MSGS)
        cache.put(key, "response")
        self.assertIs(cache.has(key), True)

    def test_has_false(self):
        cache = LLMCache()
        self.assertIs(cache.has("missing"), False)

    def test_delete(self):
        cache = LLMCache()
        key = cache.make_key(MSGS)
        cache.put(key, "r")
        self.assertIs(cache.delete(key), True)
        self.assertIsNone(cache.get(key))

    def test_delete_missing(self):
        cache = LLMCache()
        self.assertIs(cache.delete("missing"), False)


class EvictionTests(unittest.TestCase):
    def test_lru_eviction(self):
        cache = LLMCache(max_size=2)
        k1 = cache.make_key([{"role": "user", "content": "1"}])
        k2 = cache.make_key([{"role": "user", "content": "2"}])
        k3 = cache.make_key([{"role": "user", "content": "3"}])
        cache.put(k1, "r1")
        cache.put(k2, "r2")
        cache.put(k3, "r3")  # evicts k1
        self.assertEqual(cache.size, 2)
        self.assertIsNone(cache.get(k1))  # evicted
        self.assertEqual(cache.get(k2), "r2")
        self.assertEqual(cache.get(k3), "r3")

    def test_lru_access_order(self):
        cache = LLMCache(max_size=2)
        k1 = cache.make_key([{"role": "user", "content": "1"}])
        k2 = cache.make_key([{"role": "user", "content": "2"}])
        k3 = cache.make_key([{"role": "user", "content": "3"}])
        cache.put(k1, "r1")
        cache.put(k2, "r2")
        cache.get(k1)  # access k1 -> moves to MRU
        cache.put(k3, "r3")  # evicts k2 (LRU)
        self.assertIsNone(cache.get(k2))  # evicted
        self.assertEqual(cache.get(k1), "r1")

    def test_reput_existing_key_refreshes_recency(self):
        cache = LLMCache(max_size=2)
        k1 = cache.make_key([{"role": "user", "content": "1"}])
        k2 = cache.make_key([{"role": "user", "content": "2"}])
        k3 = cache.make_key([{"role": "user", "content": "3"}])
        cache.put(k1, "r1")
        cache.put(k2, "r2")
        cache.put(k1, "r1b")  # re-put k1 -> now MRU
        cache.put(k3, "r3")  # evicts k2 (LRU), not k1
        self.assertEqual(cache.get(k1), "r1b")
        self.assertIsNone(cache.get(k2))


class TTLTests(unittest.TestCase):
    def test_ttl_expiry(self):
        cache = LLMCache(default_ttl=0.05)
        key = cache.make_key(MSGS)
        cache.put(key, "response")
        time.sleep(0.1)
        self.assertIsNone(cache.get(key))

    def test_ttl_not_expired(self):
        cache = LLMCache(default_ttl=60.0)
        key = cache.make_key(MSGS)
        cache.put(key, "response")
        self.assertEqual(cache.get(key), "response")

    def test_per_put_ttl_overrides_default(self):
        cache = LLMCache(default_ttl=60.0)
        key = cache.make_key(MSGS)
        cache.put(key, "r", ttl=0.05)  # shorter than default
        time.sleep(0.1)
        self.assertIsNone(cache.get(key))

    def test_has_expired(self):
        cache = LLMCache(default_ttl=0.05)
        key = cache.make_key(MSGS)
        cache.put(key, "r")
        time.sleep(0.1)
        self.assertIs(cache.has(key), False)

    def test_prune_expired(self):
        cache = LLMCache(default_ttl=0.05)
        k1 = cache.make_key([{"role": "user", "content": "1"}])
        k2 = cache.make_key([{"role": "user", "content": "2"}])
        cache.put(k1, "r1")
        cache.put(k2, "r2", ttl=60.0)  # long TTL
        time.sleep(0.1)
        removed = cache.prune_expired()
        self.assertEqual(removed, 1)
        self.assertEqual(cache.get(k2), "r2")

    def test_prune_expired_none_removed(self):
        cache = LLMCache(default_ttl=60.0)
        cache.put(cache.make_key(MSGS), "r")
        self.assertEqual(cache.prune_expired(), 0)


class StatsTests(unittest.TestCase):
    def test_clear(self):
        cache = LLMCache()
        cache.put(cache.make_key(MSGS), "r")
        cache.clear()
        self.assertEqual(cache.size, 0)

    def test_clear_resets_stats(self):
        cache = LLMCache()
        key = cache.make_key(MSGS)
        cache.put(key, "r")
        cache.get(key)
        cache.get("miss")
        cache.clear()
        self.assertEqual(cache.stats["hits"], 0)
        self.assertEqual(cache.stats["misses"], 0)

    def test_stats(self):
        cache = LLMCache()
        key = cache.make_key(MSGS)
        cache.get("miss1")
        cache.get("miss2")
        cache.put(key, "r")
        cache.get(key)
        stats = cache.stats
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 2)
        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["max_size"], 256)

    def test_hit_rate(self):
        cache = LLMCache()
        key = cache.make_key(MSGS)
        cache.put(key, "r")
        cache.get(key)
        cache.get("miss")
        self.assertEqual(cache.hit_rate, 0.5)

    def test_hit_rate_no_requests(self):
        cache = LLMCache()
        self.assertIsNone(cache.hit_rate)

    def test_expired_get_counts_as_miss(self):
        cache = LLMCache(default_ttl=0.05)
        key = cache.make_key(MSGS)
        cache.put(key, "r")
        time.sleep(0.1)
        self.assertIsNone(cache.get(key))
        self.assertEqual(cache.stats["misses"], 1)
        self.assertEqual(cache.stats["hits"], 0)


class KeyTests(unittest.TestCase):
    def test_make_key_stable(self):
        cache = LLMCache()
        k1 = cache.make_key(MSGS, model="claude")
        k2 = cache.make_key(MSGS, model="claude")
        self.assertEqual(k1, k2)

    def test_make_key_different_model(self):
        cache = LLMCache()
        k1 = cache.make_key(MSGS, model="claude")
        k2 = cache.make_key(MSGS, model="gpt4")
        self.assertNotEqual(k1, k2)

    def test_make_key_different_messages(self):
        cache = LLMCache()
        k1 = cache.make_key([{"role": "user", "content": "a"}])
        k2 = cache.make_key([{"role": "user", "content": "b"}])
        self.assertNotEqual(k1, k2)

    def test_make_key_extras_affect_key(self):
        cache = LLMCache()
        k1 = cache.make_key(MSGS, model="m", extras={"temperature": 0.0})
        k2 = cache.make_key(MSGS, model="m", extras={"temperature": 1.0})
        self.assertNotEqual(k1, k2)

    def test_make_key_extras_order_independent(self):
        cache = LLMCache()
        k1 = cache.make_key(MSGS, extras={"a": 1, "b": 2})
        k2 = cache.make_key(MSGS, extras={"b": 2, "a": 1})
        self.assertEqual(k1, k2)

    def test_make_key_is_hex_sha256(self):
        cache = LLMCache()
        key = cache.make_key(MSGS)
        self.assertEqual(len(key), 64)
        int(key, 16)  # raises ValueError if not valid hex


class WrapTests(unittest.TestCase):
    def test_wrap_decorator(self):
        cache = LLMCache()
        calls = []

        @cache.wrap(model="test")
        def call_llm(messages):
            calls.append(1)
            return {"content": "response"}

        r1 = call_llm(MSGS)
        r2 = call_llm(MSGS)  # should be cached
        self.assertEqual(r1, r2)
        self.assertEqual(len(calls), 1)  # only called once

    def test_wrap_caches_none_result(self):
        """A None (or falsy) return value must still be cached, not recomputed."""
        cache = LLMCache()
        calls = []

        @cache.wrap(model="test")
        def call_llm(messages):
            calls.append(1)
            return None

        self.assertIsNone(call_llm(MSGS))
        self.assertIsNone(call_llm(MSGS))  # served from cache, not recomputed
        self.assertEqual(len(calls), 1)

    def test_wrap_preserves_function_metadata(self):
        cache = LLMCache()

        @cache.wrap()
        def call_llm(messages):
            """Doc string."""
            return "x"

        self.assertEqual(call_llm.__name__, "call_llm")
        self.assertEqual(call_llm.__doc__, "Doc string.")

    def test_wrap_passes_through_extra_args(self):
        cache = LLMCache()
        seen = []

        @cache.wrap(model="test")
        def call_llm(messages, suffix=""):
            seen.append(suffix)
            return f"resp{suffix}"

        self.assertEqual(call_llm(MSGS, suffix="!"), "resp!")
        # Same messages -> cached, so the function body (and suffix) is not re-run.
        self.assertEqual(call_llm(MSGS, suffix="?"), "resp!")
        self.assertEqual(seen, ["!"])


class GetDefaultTests(unittest.TestCase):
    def test_get_default_returned_on_miss(self):
        cache = LLMCache()
        sentinel = object()
        self.assertIs(cache.get("missing", default=sentinel), sentinel)

    def test_get_cached_falsy_value(self):
        cache = LLMCache()
        key = cache.make_key(MSGS)
        cache.put(key, None)
        sentinel = object()
        self.assertIsNone(cache.get(key, default=sentinel))  # real hit, not default


class ValidationTests(unittest.TestCase):
    def test_invalid_max_size(self):
        with self.assertRaises(ValueError):
            LLMCache(max_size=0)

    def test_invalid_default_ttl(self):
        with self.assertRaises(ValueError):
            LLMCache(default_ttl=0)

    def test_negative_default_ttl(self):
        with self.assertRaises(ValueError):
            LLMCache(default_ttl=-5.0)


class CacheEntryTests(unittest.TestCase):
    def test_cache_entry_expired_property(self):
        entry = CacheEntry(key="k", response="r", ttl=None)
        self.assertIs(entry.expired, False)
        expired = CacheEntry(
            key="k", response="r", created_at=time.time() - 10, ttl=1.0
        )
        self.assertIs(expired.expired, True)

    def test_cache_entry_hit_count_increments(self):
        cache = LLMCache()
        key = cache.make_key(MSGS)
        cache.put(key, "r")
        cache.get(key)
        cache.get(key)
        with cache._lock:
            entry = cache._store[key]
        self.assertEqual(entry.hit_count, 2)


if __name__ == "__main__":
    unittest.main()
