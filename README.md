# Loom

Thread-and-episode orchestration on its own minimal engine (any-llm providers).

Orchestrator dispatches **threads**; each thread runs a bounded action in a worker context and returns an **episode**. Episodes are the only thing that crosses between threads.

```sh
uv tool install loom-threads
loom setup  # provider config (once)
loom "add rate limiting to the API"
```

From source:

```sh
uv sync
uv run loom "add rate limiting to the API"
```

Docs: [wiki/quickstart.md](wiki/quickstart.md)
