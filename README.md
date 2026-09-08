# Loom

Thread-and-episode orchestration, built on [Tau](https://github.com/huggingface/tau) as the
agent engine.

An orchestrator plans and decomposes but cannot touch files: it only dispatches **threads**.
Each thread runs a bounded action in a fresh worker context and returns an **episode** — a
compressed record of what it did. Episodes are the only thing that crosses between threads.

```text
orchestrator (thread tools only)
  └─ thread(name="impl/auth", action="add JWT middleware", threads=["research"])
       └─ worker (Tau coding tools: read/write/edit/bash)
            → final response → episodes.jsonl → tool result
```

## Install

```sh
uv sync            # resolves tau-ai from ../tau (editable)
```

## Run

```sh
uv run loom "add rate limiting to the API"
```

Options: `--provider`, `--model`, `--cwd`, `--store`, `--max-turns`.

Episodes append to `<cwd>/.loom/episodes.jsonl`.

## Layout

| module | role |
| --- | --- |
| `loom/engine.py` | Tau bootstrap: provider, model, worker coding tools |
| `loom/episodes.py` | `Episode` + append-only JSONL store + renderers |
| `loom/dispatch.py` | one dispatch: build worker context, run, store episode |
| `loom/threads.py` | the orchestrator's toolset (`thread`, `threads`, `thread_read`) |
| `loom/prompts.py` | orchestrator / worker system prompts |
| `loom/cli.py` | print-mode orchestrator loop |

## What Tau owns

Streaming, retries, provider config, credentials, the agent loop, events, and tool safety.
Loom adds ~400 lines: the episode store, the dispatch function, and three tools.

## Not implemented yet

Batched parallel dispatch with dependency ordering (nac's DAG), steering in-flight threads,
subprocess-isolated workers, TUI.
