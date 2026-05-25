# llm-cache-mem-py

In-process LRU cache for LLM responses. Keyed by (messages, model, extras) with TTL support.

## Install

```bash
pip install llm-cache-mem-py
```

## Usage

```python
from llm_cache_mem import LLMCache

cache = LLMCache(max_size=100, default_ttl=300.0)  # 5-minute TTL

# Manual
key = cache.make_key(messages, model="claude-sonnet")
if cache.has(key):
    return cache.get(key)
response = client.chat(messages)
cache.put(key, response)

# Decorator
@cache.wrap(model="claude-sonnet", ttl=600.0)
def call_llm(messages):
    return client.chat(messages)

# Stats
print(cache.stats)   # hits, misses, hit_rate, size
cache.prune_expired()
cache.clear()
```

## License

MIT
