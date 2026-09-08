# Loom

Thread-and-episode orchestration on its own minimal engine (any-llm providers).

An orchestrator plans and decomposes but cannot touch files: it only dispatches **threads**.
Each thread runs a bounded action in a fresh worker context and returns an **episode** — a
compressed record of what it did. Episodes are the only thing that crosses between threads.

```text
orchestrator (thread tools only)
  └─ thread(name="impl/auth", action="add JWT middleware", threads=["research"])
       (or thread_batch(items=[...]) to run independent threads concurrently)
       └─ worker (loom coding tools: read/write/edit/bash)
            → final response → episodes.jsonl → tool result
```

## Install

```sh
uv sync
```

## Run

```sh
uv run loom "add rate limiting to the API"
```

Options: `--provider`, `--model` (`provider:model` or plain), `--cwd`, `--store`, `--max-turns`.

## Provider config

```sh
uv run loom setup
```

Prompts for base URL, API key, model (`provider:model`), and optional
provider override. Saves to `~/.loom/credential.json` (mode 600).
Works with any OpenAI-compatible endpoint (vLLM, Ollama, custom gateway).

Precedence: flags > `LOOM_*` env (`LOOM_PROVIDER`, `LOOM_MODEL`,
`LOOM_LLM_PROVIDER_API_KEY`, `LOOM_LLM_PROVIDER_BASE_URL`) > credential
file > default.

Episodes append to `<cwd>/.loom/episodes.jsonl`.

## Layout

| module | role |
| --- | --- |
| `loom/agent.py` | minimal loop on any-llm: `Tool`, `run_loop`, 5 events |
| `loom/coding.py` | worker tools: read/write/edit/bash (cwd-jailed, truncated) |
| `loom/engine.py` | provider/model resolve + worker toolset |
| `loom/episodes.py` | `Episode` + append-only JSONL store + renderers |
| `loom/dispatch.py` | one dispatch: build worker context, run, store episode |
| `loom/threads.py` | the orchestrator's toolset (`thread`, `thread_batch`, `threads`, `thread_read`) |
| `loom/prompts.py` | orchestrator / worker system prompts |
| `loom/cli.py` | print-mode orchestrator loop |

## What the engine owns

any-llm handles providers, auth, and retries. Loom owns the loop (~120 lines),
4 coding tools, the episode store, dispatch, and 4 thread tools.

## Not implemented yet

Steering in-flight threads, subprocess-isolated workers, TUI.

`thread_batch` dispatches several threads as one batch: items with no in-batch source
dependency run concurrently, and an item naming another batch item as a source waits for it.
