# Operations

## Provider precedence

`build_engine(provider_name, model, cwd)` in `src/loom/engine.py`. Credential file values are applied via `os.environ.setdefault` — **a file never overwrites the process env** (asserted in `tests/test_config.py`).

1. CLI flags `--provider` / `--model`
2. Env `LOOM_PROVIDER`, `LOOM_MODEL`, `LOOM_LLM_PROVIDER_API_KEY`, `LOOM_LLM_PROVIDER_BASE_URL`
3. `~/.loom/credential.json` (`LOOM_CREDENTIAL_FILE` overrides the path in tests), keys `base_url / api_key / model / provider`
4. Default: provider `None` (any-llm default), model `openai:gpt-5.4`

`loom setup` writes the file as plain JSON, mode 600, creating parents; corrupt/missing file reads as `{}`. `split_model` (`src/loom/agent.py`) accepts `provider:model` or a plain id.

## Sessions and resume

- Each run appends to `<cwd>/.loom/sessions/<id>.jsonl`; each worker result appends to `<cwd>/.loom/episodes/<id>.jsonl`. Both dirs are gitignored (`.loom/`).
- `--resume <id>` once + existing file → appends the new prompt to that file and prepends its rendered transcript as context. Multiple `--resume` ids (or an unknown id) → **new** session file with all found transcripts prepended; missing ids print `warning: session '<id>' not found` to stderr.
- CLI prints `session: <path>` at the end of every run.
- Episode lines in sessions are id references; use `SessionStore.render(sid, episode_store)` to resolve content (see `src/loom/sessions.py`, `tests/test_sessions.py`).

## Threading patterns that work

From `src/loom/prompts.py` + `src/loom/threads.py` semantics:

- `thread(name="research", action="...")` first when the area/failure mode is unclear; then `thread(name="impl/<topic>", action="...", threads=["research"])`.
- `thread_batch` for independent explorations (best-of-N), then a `synth` item with `threads: ["a","b"]` to combine. Only batch threads touching different files — concurrent workers share one `cwd` and can overwrite each other.
- Reuse a thread name to continue a stream (worker sees full retained history); new name for a distinct workstream. Read back with `threads()` / `thread_read(name)`.
- Before declaring done, dispatch a fresh `verify/<topic>` thread rather than trusting the impl thread's judgment.
- Keep dispatches to one concrete action with a dense episode; weak episodes force later threads to rediscover setup/verification state.

## Testing

```sh
uv run pytest        # 3 files: test_config, test_dispatch, test_sessions
uv run ruff check .  # line-length 100, target py312; prompts.py exempt from E501
uv run ty check   # `ty` is the type checker
```

`tests/test_dispatch.py` defines the reusable pattern: a scripted `FakeLLM` (`text` or `{tool_calls:[...]}` items) monkeypatched over `any_llm.AnyLLM.create`, then `asyncio.run(run_dispatch(...))` or `batch.execute({...})` against an `EpisodeStore` in `tmp_path`. Timing-sensitive test: `test_batch_runs_independent_items_concurrently` asserts elapsed < 0.09s for two 0.05s sleeps.

## Where to change code

| want | touch | watch for |
| --- | --- | --- |
| new worker capability | `src/loom/coding.py:create_coding_tools` + `src/loom/engine.py:build_engine` | keep cwd jail; `edit` exact-once matching; truncation limits |
| new orchestrator tool | `src/loom/threads.py:create_thread_tools` + prompt tool list in `src/loom/prompts.py` | `details.episodes` shape `cli.py:_episode_entries` parses |
| loop semantics | `src/loom/agent.py:run_loop` | sequential calls; `messages` mutated in place; lazy `any_llm` import |
| context shaping | `src/loom/dispatch.py:run_dispatch`, renderers in `src/loom/episodes.py` | ok-only filtering; latest-only sources |
| CLI/output | `src/loom/cli.py` | `_preview` truncation, `<< name (N chars)` lines, `session:` footer |
| provider/config | `src/loom/engine.py` | setdefault ordering; `LOOM_CREDENTIAL_FILE` override |

## Not implemented

Steering in-flight threads, subprocess-isolated workers, TUI (README). Concurrent same-name dispatch is rejected, not queued.
