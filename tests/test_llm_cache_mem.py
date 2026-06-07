"""Tests for llm-cache-mem-py."""

import time
import pytest
from llm_cache_mem import LLMCache, CacheEntry


MSGS = [{"role": "user", "content": "Hello"}]


def test_put_and_get():
    cache = LLMCache()
    key = cache.make_key(MSGS)
    cache.put(key, {"content": "Hi!"})
    assert cache.get(key) == {"content": "Hi!"}


def test_get_miss():
    cache = LLMCache()
    assert cache.get("nonexistent") is None


def test_has_true():
    cache = LLMCache()
    key = cache.make_key(MSGS)
    cache.put(key, "response")
    assert cache.has(key) is True


def test_has_false():
    cache = LLMCache()
    assert cache.has("missing") is False


def test_delete():
    cache = LLMCache()
    key = cache.make_key(MSGS)
    cache.put(key, "r")
    assert cache.delete(key) is True
    assert cache.get(key) is None


def test_delete_missing():
    cache = LLMCache()
    assert cache.delete("missing") is False


def test_lru_eviction():
    cache = LLMCache(max_size=2)
    k1 = cache.make_key([{"role": "user", "content": "1"}])
    k2 = cache.make_key([{"role": "user", "content": "2"}])
    k3 = cache.make_key([{"role": "user", "content": "3"}])
    cache.put(k1, "r1")
    cache.put(k2, "r2")
    cache.put(k3, "r3")  # evicts k1
    assert cache.size == 2
    assert cache.get(k1) is None  # evicted
    assert cache.get(k2) == "r2"
    assert cache.get(k3) == "r3"


def test_lru_access_order():
    cache = LLMCache(max_size=2)
    k1 = cache.make_key([{"role": "user", "content": "1"}])
    k2 = cache.make_key([{"role": "user", "content": "2"}])
    k3 = cache.make_key([{"role": "user", "content": "3"}])
    cache.put(k1, "r1")
    cache.put(k2, "r2")
    cache.get(k1)  # access k1 → moves to MRU
    cache.put(k3, "r3")  # evicts k2 (LRU)
    assert cache.get(k2) is None  # evicted
    assert cache.get(k1) == "r1"


def test_ttl_expiry():
    cache = LLMCache(default_ttl=0.05)
    key = cache.make_key(MSGS)
    cache.put(key, "response")
    time.sleep(0.1)
    assert cache.get(key) is None


def test_ttl_not_expired():
    cache = LLMCache(default_ttl=60.0)
    key = cache.make_key(MSGS)
    cache.put(key, "response")
    assert cache.get(key) == "response"


def test_has_expired():
    cache = LLMCache(default_ttl=0.05)
    key = cache.make_key(MSGS)
    cache.put(key, "r")
    time.sleep(0.1)
    assert cache.has(key) is False


def test_prune_expired():
    cache = LLMCache(default_ttl=0.05)
    k1 = cache.make_key([{"role": "user", "content": "1"}])
    k2 = cache.make_key([{"role": "user", "content": "2"}])
    cache.put(k1, "r1")
    cache.put(k2, "r2", ttl=60.0)  # long TTL
    time.sleep(0.1)
    removed = cache.prune_expired()
    assert removed == 1
    assert cache.get(k2) == "r2"


def test_clear():
    cache = LLMCache()
    cache.put(cache.make_key(MSGS), "r")
    cache.clear()
    assert cache.size == 0


def test_stats():
    cache = LLMCache()
    key = cache.make_key(MSGS)
    cache.get("miss1")
    cache.get("miss2")
    cache.put(key, "r")
    cache.get(key)
    stats = cache.stats
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["size"] == 1


def test_hit_rate():
    cache = LLMCache()
    key = cache.make_key(MSGS)
    cache.put(key, "r")
    cache.get(key)
    cache.get("miss")
    assert cache.hit_rate == 0.5


def test_hit_rate_no_requests():
    cache = LLMCache()
    assert cache.hit_rate is None


def test_make_key_stable():
    cache = LLMCache()
    k1 = cache.make_key(MSGS, model="claude")
    k2 = cache.make_key(MSGS, model="claude")
    assert k1 == k2


def test_make_key_different_model():
    cache = LLMCache()
    k1 = cache.make_key(MSGS, model="claude")
    k2 = cache.make_key(MSGS, model="gpt4")
    assert k1 != k2


def test_wrap_decorator():
    cache = LLMCache()
    calls = []

    @cache.wrap(model="test")
    def call_llm(messages):
        calls.append(1)
        return {"content": "response"}

    r1 = call_llm(MSGS)
    r2 = call_llm(MSGS)  # should be cached
    assert r1 == r2
    assert len(calls) == 1  # only called once


def test_wrap_caches_none_result():
    """A None (or falsy) return value must still be cached, not recomputed."""
    cache = LLMCache()
    calls = []

    @cache.wrap(model="test")
    def call_llm(messages):
        calls.append(1)
        return None

    assert call_llm(MSGS) is None
    assert call_llm(MSGS) is None  # served from cache, not recomputed
    assert len(calls) == 1


def test_get_default_returned_on_miss():
    cache = LLMCache()
    sentinel = object()
    assert cache.get("missing", default=sentinel) is sentinel


def test_get_cached_falsy_value():
    cache = LLMCache()
    key = cache.make_key(MSGS)
    cache.put(key, None)
    sentinel = object()
    assert cache.get(key, default=sentinel) is None  # real hit, not the default


def test_invalid_max_size():
    with pytest.raises(ValueError):
        LLMCache(max_size=0)


def test_invalid_default_ttl():
    with pytest.raises(ValueError):
        LLMCache(default_ttl=0)


def test_cache_entry_expired_property():
    entry = CacheEntry(key="k", response="r", ttl=None)
    assert entry.expired is False
    expired = CacheEntry(key="k", response="r", created_at=time.time() - 10, ttl=1.0)
    assert expired.expired is True
