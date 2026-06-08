# llm-cache-mem-py

In-process, thread-safe **LRU cache for LLM responses**, keyed by
`(messages, model, extras)` with optional per-entry TTL support.

LLM calls are slow and expensive. When your app issues the *same* request more
than once — retries, repeated prompts, idempotent tool calls, test runs — this
library returns the previous response from memory instead of hitting the API
again. It is a single dependency-free module: no Redis, no disk, no network.

## Features

- **LRU eviction** with a configurable `max_size`.
- **Optional TTL** per cache (a `default_ttl`) and per entry (`put(..., ttl=...)`).
- **Stable content-addressed keys**: a SHA-256 hash of the JSON-serialized
  `(messages, model, extras)`, so equal requests map to the same key regardless
  of dict ordering.
- **Thread-safe**: every mutation is guarded by a lock, so the cache can be
  shared across worker threads.
- **Correctly caches falsy responses** (`None`, `""`, `0`, `[]`) — a cached
  `None` is a hit, not a miss.
- **Hit/miss statistics** for measuring effectiveness.
- **Zero runtime dependencies**; ships with type hints and a `py.typed` marker.

## Install

```bash
pip install llm-cache-mem-py
```

Or from source:

```bash
git clone https://github.com/MukundaKatta/llm-cache-mem-py
pip install ./llm-cache-mem-py
```

## Usage

### Manual get / put

```python
from llm_cache_mem import LLMCache

cache = LLMCache(max_size=100, default_ttl=300.0)  # 5-minute TTL

def call_llm(messages):
    key = cache.make_key(messages, model="claude-sonnet")
    if cache.has(key):
        return cache.get(key)
    response = client.chat(messages)   # your real LLM call
    cache.put(key, response)
    return response
```

### As a decorator

`wrap` keys the cache on the first positional argument (the messages) plus the
`model`/`extras` you pass to the decorator:

```python
@cache.wrap(model="claude-sonnet", ttl=600.0)
def call_llm(messages):
    return client.chat(messages)

call_llm(messages)  # miss -> calls the API, stores result
call_llm(messages)  # hit  -> returns the stored result, API not called
```

### Including sampling params in the key

Anything that changes the response should be part of the key. Pass it via
`extras` so two requests that differ only in, say, `temperature` are cached
separately:

```python
key = cache.make_key(
    messages,
    model="claude-sonnet",
    extras={"temperature": 0.7, "max_tokens": 512},
)
```

### Inspecting and maintaining the cache

```python
print(cache.stats)     # {'size': 1, 'max_size': 100, 'hits': 3, 'misses': 1, 'hit_rate': 0.75}
print(cache.hit_rate)  # 0.75 (None until the first get)
cache.prune_expired()  # drop expired entries, returns how many were removed
cache.clear()          # empty the cache and reset stats
```

## API

### `LLMCache(max_size=256, default_ttl=None)`

Create a cache. `max_size` must be `>= 1`. `default_ttl` (seconds) must be a
positive number or `None` (no expiry). Both invalid values raise `ValueError`.

| Method / property | Description |
| --- | --- |
| `make_key(messages, model=None, extras=None) -> str` | Compute the stable SHA-256 cache key for a request. |
| `has(key) -> bool` | `True` if the key exists and is not expired (expired entries are dropped). |
| `get(key, default=None) -> Any` | Return the cached response, or `default` on miss/expiry. Records a hit or miss. |
| `put(key, response, ttl=None) -> None` | Store a response. `ttl` overrides `default_ttl`. Evicts the LRU entry when full. |
| `delete(key) -> bool` | Remove an entry; `True` if it existed. |
| `clear() -> None` | Empty the cache and reset hit/miss counters. |
| `prune_expired() -> int` | Remove all expired entries; returns the count removed. |
| `wrap(model=None, extras=None, ttl=None)` | Decorator that memoizes a function keyed on its first positional argument. |
| `size` (property) | Current number of entries. |
| `hit_rate` (property) | `hits / (hits + misses)`, or `None` before any `get`. |
| `stats` (property) | `dict` with `size`, `max_size`, `hits`, `misses`, `hit_rate`. |

### `CacheEntry`

The dataclass stored internally for each response: `key`, `response`,
`created_at`, `ttl`, `hit_count`, and an `expired` property.

## Development

Run the test suite (standard library only, no third-party dependencies):

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT
