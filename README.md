# Loom

Loom (`loom-threads`) is a **thread-and-episode orchestration** agent harness on its own minimal
agent engine over `any-llm` providers. An **orchestrator** LLM plans and decomposes work but is
deliberately tool-restricted: it can only dispatch **threads**. Each thread runs one bounded action
in a fresh worker context (file/bash tools, jailed to a working directory) and returns an
**episode** — a compressed record of what was done. Episodes are the only data that crosses between
threads; nothing else leaks into orchestrator or worker context.

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
