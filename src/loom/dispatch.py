"""One thread dispatch: build a fresh worker context, run it, keep the episode."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from loom.agent import AgentError, AssistantEnd, Event, Tool, run_loop
from loom.episodes import (
    CANCELLED,
    ERROR,
    OK,
    TIMED_OUT,
    Episode,
    EpisodeStore,
    render_self_context,
    render_source_context,
)
from loom.prompts import worker_prompt

DEFAULT_TIMEOUT_SECS = 1800.0
DEFAULT_MAX_TURNS = 64

WorkerListener = Callable[[str, Event], None]


async def run_dispatch(
    *,
    provider: str | None,
    model: str,
    worker_tools: Sequence[Tool],
    store: EpisodeStore,
    name: str,
    action: str,
    source_threads: Sequence[str] = (),
    working_directory: str | Path = ".",
    timeout_secs: float = DEFAULT_TIMEOUT_SECS,
    max_turns: int = DEFAULT_MAX_TURNS,
    on_event: WorkerListener | None = None,
) -> str:
    """Run one action in a worker and return the episode it handed back.

    The worker only ever sees: the worker prompt, this thread's own retained
    episodes, the latest episode of each source thread, and the action. Its
    final response is stored as the next episode for `name` and returned.
    """
    system = worker_prompt(str(working_directory))

    messages: list[dict[str, Any]] = []
    own = store.read(name)
    if own:
        messages.append({"role": "user", "content": render_self_context(name, own)})
    for source in source_threads:
        episode = store.latest(source)
        if episode is None:
            return f"Error: source thread '{source}' has no retained episode."
        messages.append({"role": "user", "content": render_source_context(episode)})
    messages.append({"role": "user", "content": action})

    final = ""
    error = ""

    async def consume() -> None:
        nonlocal final, error
        async for event in run_loop(
            provider=provider,
            model=model,
            system=system,
            messages=messages,
            tools=list(worker_tools),
            max_turns=max_turns,
        ):
            if on_event is not None:
                on_event(name, event)
            if isinstance(event, AssistantEnd) and event.text.strip():
                final = event.text.strip()
            elif isinstance(event, AgentError):
                error = event.message

    try:
        await asyncio.wait_for(consume(), timeout=timeout_secs)
    except TimeoutError:
        store.append(Episode(name, action, "", TIMED_OUT))
        return f"Error: thread '{name}' timed out after {int(timeout_secs)}s."
    except asyncio.CancelledError:
        store.append(Episode(name, action, "", CANCELLED))
        raise

    if not final:
        store.append(Episode(name, action, "", ERROR))
        detail = f": {error}" if error else ""
        return f"Error: thread '{name}' produced no episode{detail}"

    store.append(Episode(name, action, final, OK))
    return final
