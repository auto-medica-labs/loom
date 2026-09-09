# Architecture

Why this shape exists: a single long-lived agent accumulates noisy context and loses decisions between turns. Loom instead forces all work through bounded worker dispatches whose **final response is stored as an episode** — a compressed handoff future dispatches can read. The orchestrator never touches files; workers never talk to each other directly. Episodes are the synchronization primitive.

Execution flow for one `uv run loom "..."` (`src/loom/cli.py:_run`):

1. `build_engine()` resolves provider/model/worker tools (`src/loom/engine.py`).
2. `SessionStore.start()` opens `<cwd>/.loom/sessions/<stamp-rand>.jsonl`; `--session` ids and `--resume` render into the first messages.
3. `create_thread_tools(...)` builds the orchestrator's 4 tools.
4. `run_loop(...)` with `orchestrator_prompt` runs until no more tool calls or `max_turns=32`.
5. Each tool result's per-episode `(name, text, id)` triples are logged as `episode` references in the session; plain text as `output`.

## Agent loop — `src/loom/agent.py`

ReAct loop on any-llm. Key facts:

- `Tool { name, description, parameters, execute_fn }`; `execute()` catches all exceptions into `ToolResult(text="Error: ...", is_error=True)` — tools are an isolation boundary.
- `run_loop(provider, model, system, messages, tools, max_turns, api_key, api_base)` mutates `messages` in place (assistant + tool turns appended). Yields 5 events: `TextDelta | ToolStart | ToolEnd | AssistantEnd | AgentError`.
- Tool calls run **sequentially** in the order the model emitted them. Malformed JSON args become `{}`; unknown tool name becomes an error result, not a crash.
- `split_model("provider:model")` splits on the first `:`; plain ids fall back to `provider` arg or `"openai"`.
- `AnyLLM.create(provider, api_key, api_base)` is imported lazily so tests stay cheap. Provider errors surface as `AgentError`.
- any-llm owns providers/auth/retries; Loom owns the loop, 4 coding tools, episode store, dispatch, 4 thread tools.

## Worker coding tools — `src/loom/coding.py`

`create_coding_tools(cwd)` returns `read / write / edit / bash`.

- **Jail:** `_resolve()` rejects any path escaping `cwd` — security, never simplify away.
- **Truncation:** last 2000 lines or last 50KB; `read` supports `offset`/`limit`.
- `write` creates parents, overwrites. `edit` requires each `oldText` to match **exactly once** (`matches Nx (need 1)` otherwise) and applies atomically under a per-path `asyncio.Lock`.
- `bash` runs via `create_subprocess_shell` in `cwd` with `DEVNULL` stdin, merged stderr, default 120s timeout (`timeout` arg must be > 0). Non-zero exit returns `is_error=True` with `[exit N]` suffix.
- All tools are text-only (UTF-8); binary reads error.

## One dispatch — `src/loom/dispatch.py`

`run_dispatch(provider, model, worker_tools, store, name, action, session, source_threads, working_directory, timeout_secs=1800, max_turns=64, on_event)`:

1. Builds worker messages: thread's **own retained episodes** (`render_self_context`, full history) + **latest episode of each named source** (`render_source_context`, latest only) + the action. Missing source → immediate `Error: source thread '...' has no retained episode.` with nothing stored.
2. Runs `run_loop` with `worker_prompt` via `asyncio.wait_for(consume(), timeout)`. Last non-empty `AssistantEnd` wins as `final`.
3. Stores one `Episode(name, action, final, status, session)`: `ok` on success; `timed_out` / `cancelled` / `error` (empty final) otherwise — and returns the text or an `Error: ...` string. Timeout message: `Error: thread '<name>' timed out after <N>s.`
4. `on_event(name, event)` mirrors worker `ToolStart/ToolEnd` lines to the CLI (`    [<name>] ...`).

Failure episodes are stored but **excluded from future context** (`EpisodeStore.read` filters `ok_only=True`).

## Orchestrator thread tools — `src/loom/threads.py`

`create_thread_tools(provider, model, worker_tools, store, working_directory, timeout_secs, on_event, session)` returns:

| tool | behavior |
| --- | --- |
| `thread(name, action, threads?, timeout?)` | one `run_dispatch`; needs `name` + `action`; rejects a second concurrent dispatch of the same name (`already running`) |
| `thread_batch(items[])` | waves of concurrent `run_dispatch` via `asyncio.gather`; dependent items wait for in-batch sources and receive their episodes |
| `threads()` | `Active threads:\n- <name> \| <n> episodes` or `No active threads in this session.` |
| `thread_read(name)` | full retained history via `render_thread_document` |

`thread`/`thread_batch` results carry `details.episodes = [{name, text, id}]` so `cli.py` can log episode references per dispatch in order. Failed dispatches get `id=None`.

**Batch planning — `plan_waves(names, sources)`:** topological waves over in-batch edges only (sources naming earlier-turn threads need no ordering — they're already in the store). Raises `ValueError` on duplicate names, self-dependency, or cycles, in which case **nothing runs**. A dependent whose source failed is skipped with `Error: source thread '<dead>' failed; dispatch '<name>' skipped.` Wave semantics are covered by `tests/test_dispatch.py` (`plan_waves_orders...`, `ignores_sources_outside_the_batch`, `rejects_duplicate/cycles`, `runs_independent_items_concurrently`, `dependent_item_sees_its_sources`, `skips_dependent_when_source_fails`).

## Episodes — `src/loom/episodes.py`

`Episode { thread, action, content, status=ok, created_at, id (12 hex), session }`. `content` is the worker's final response verbatim.

`EpisodeStore(path)`: one file per episode at `<dir>/<id>.jsonl`. API: `append` (collision-safe), `read(thread)` (ok-only), `latest(thread)`, `names()`, `count(thread)`, `get(id)`, `by_session(id)` (failures included). Ordering is by `(mtime_ns, name)`. Corrupt lines are skipped.

Renderers: `render_self_context` (numbered `=== Episode N | ... ===` history), `render_source_context` (latest-of-source), `render_thread_document` (for `thread_read`).

## Sessions — `src/loom/sessions.py`

Orchestrator-side transcript; episodes keep worker results, sessions keep the orchestrator's side. One `<id>.jsonl` per run in `<cwd>/.loom/sessions/`, one `{type, ...}` line per turn:

- `{"type":"input","text"}` — prompt (incl. each `--session` / `--resume` follow-up)
- `{"type":"output","text","label"?}` — orchestrator text (`label` = dispatch label for tool outputs)
- `{"type":"episode","label","id"}` — **reference only**, no content (`render(sid, store)` resolves ids to `== <label> ==\n<content>`)

`render()` skips corrupt lines; unknown id returns `None`. `SessionStore.start()` ids look like `20260909-035154-562836`.

## Prompts — `src/loom/prompts.py`

Ported from `nac-core` agent prompts, trimmed for a single-model orchestrator. Orchestrator prompt encodes the threading discipline: reuse threads for ongoing streams, new thread per distinct workstream, one concrete action per dispatch, `threads` sources only when relevant, research-before-implementation when unclear, stable roles (`setup`, `impl/<topic>`, `verify/<topic>`), `thread_batch` for concurrent read-only/different-file work with best-of-N + synthesis patterns, verify-with-a-fresh-thread before declaring done, no extra Markdown files unless asked. Worker prompt defines the episode as a handoff: end goal, approach, steps, blockers, results, paths, decisions, verification, state, next follow-up — exact commands and known-good vs known-broken when relevant.
