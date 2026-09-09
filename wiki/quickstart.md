# Loom wiki — quickstart

Loom is thread-and-episode orchestration on its own minimal any-llm engine. An **orchestrator** plans and decomposes but cannot touch files — it only dispatches **threads**. Each thread runs a bounded action in a fresh worker context and returns an **episode**, a compressed record of what it did. Episodes are the only thing that crosses between threads.

```text
orchestrator (thread tools only)
  └─ thread(name="impl/auth", action="add JWT middleware", threads=["research"])
       (or thread_batch(items=[...]) for concurrent threads)
       └─ worker (read/write/edit/bash) → episodes store → tool result
```

Canonical details live in [architecture](architecture.md); setup/ops in [operations](operations.md).

## Install and run

```sh
uv sync
uv run loom "add rate limiting to the API"
```

Requirements: Python >=3.12, `any-llm-sdk` + `rich` (see `pyproject.toml`). Entry point is `loom = "loom.cli:main"`.

Useful flags (`src/loom/cli.py:_parse_args`):

| flag | default | meaning |
| --- | --- | --- |
| `--provider` | credential/env | any-llm provider name |
| `--model` | `openai:gpt-5.4` | `provider:model` or plain id |
| `--cwd` | `.` | working directory workers are jailed to |
| `--store` | `<cwd>/.loom/episodes` | episode directory |
| `--max-turns` | `32` | orchestrator loop turns (workers get 64) |
| `--resume <id>` | — | one id continues that session file; several ids start a new run with all rendered as context |
| `setup` | — | `uv run loom setup` — interactive provider config (not a flag, a subcommand) |

## Provider setup

```sh
uv run loom setup
```

Prompts for base URL, API key, model (`provider:model`), optional provider override. Saves to `~/.loom/credential.json` (mode 600). Works with any OpenAI-compatible endpoint (vLLM, Ollama, custom gateway).

Precedence (flags > env > credential file > default) — see [operations](operations.md#provider-precedence):

- Flags `--provider` / `--model`
- Env `LOOM_PROVIDER`, `LOOM_MODEL`, `LOOM_LLM_PROVIDER_API_KEY`, `LOOM_LLM_PROVIDER_BASE_URL`
- `~/.loom/credential.json` (overridable via `LOOM_CREDENTIAL_FILE` in tests)
- Default model `openai:gpt-5.4` (`src/loom/engine.py:DEFAULT_MODEL`)

## Where things live

| path | what |
| --- | --- |
| `src/loom/agent.py` | minimal any-llm loop: `Tool`, `run_loop`, 5 events |
| `src/loom/coding.py` | worker tools: read/write/edit/bash (cwd-jailed) |
| `src/loom/engine.py` | provider/model resolve + worker toolset |
| `src/loom/episodes.py` | `Episode` + one-file-per-episode store |
| `src/loom/sessions.py` | orchestrator transcript store (one file per run) |
| `src/loom/dispatch.py` | one dispatch: build worker context, run, store episode |
| `src/loom/threads.py` | orchestrator toolset: `thread`, `thread_batch`, `threads`, `thread_read` |
| `src/loom/prompts.py` | orchestrator / worker system prompts |
| `src/loom/cli.py` | print-mode orchestrator loop + `setup` |
| `<cwd>/.loom/episodes/<id>.jsonl` | one file per episode |
| `<cwd>/.loom/sessions/<session-id>.jsonl` | one file per orchestrator run |

`.loom/` is gitignored. `uv run loom --resume <session-id> "follow-up"` appends to the same session file.

## Next

- [architecture](architecture.md) — why threads/episodes exist and how each module works
- [operations](operations.md) — credentials, sessions/resume, testing, extension points
