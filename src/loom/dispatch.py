"""One thread dispatch: build a fresh worker context, run it, keep the episode."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from loom.agent import (
    Agent,
    AgentError,
    AssistantEnd,
    Event,
    RetryAttempt,
    TextDelta,
    Tool,
    ToolEnd,
    ToolStart,
)
from loom.episodes import (
    CANCELLED,
    ERROR,
    OK,
    TIMED_OUT,
    Episode,
    EpisodeStore,
    own_history_messages,
    source_message,
)
from loom.prompts import worker_prompt

DEFAULT_TIMEOUT_SECS = 1800.0
DEFAULT_MAX_TURNS = 64

WorkerListener = Callable[[str, Event], None]


def _event_to_dict(event: Event) -> dict[str, Any]:
    """Minute interaction as JSON-safe dicts for `<id>.trace.jsonl`."""
    if isinstance(event, TextDelta):
        return {"type": "text", "delta": event.delta}
    if isinstance(event, ToolStart):
        return {
            "type": "tool_start",
            "tool": event.tool_name,
            "args": event.args,
            "call_id": event.call_id,
        }
    if isinstance(event, ToolEnd):
        return {
            "type": "tool_end",
            "tool": event.tool_name,
            "text": event.result.text,
            "is_error": event.is_error,
            "call_id": event.call_id,
        }
    if isinstance(event, AssistantEnd):
        return {"type": "assistant", "text": event.text, "tool_calls": event.tool_calls}
    if isinstance(event, RetryAttempt):
        return {"type": "retry", "attempt": event.attempt, "message": event.message}
    return {"type": "error", "message": event.message}


class Dispatcher:
    """Runs one action in a worker and returns the episode it handed back.

    Holds everything a dispatch needs (agent, worker tools, episode store,
    session, jail, defaults, listener) so callers pass only the action.

    The worker only ever sees: the worker prompt, its own past as proper
    `user` (action) / `assistant` (episode) turns, the latest episode of
    each source thread as orchestrator input, and the action. Its final
    response is stored as the next episode for `name` and returned as
    `(text, episode_id)`; `episode_id` is `None` when no episode was
    retained (a failed dispatch).
    """

    def __init__(
        self,
        *,
        agent: Agent,
        worker_tools: Sequence[Tool],
        store: EpisodeStore,
        session: str,
        working_directory: str | Path = ".",
        timeout_secs: float = DEFAULT_TIMEOUT_SECS,
        max_turns: int = DEFAULT_MAX_TURNS,
        on_event: WorkerListener | None = None,
    ) -> None:
        if not session:
            raise ValueError("Dispatcher requires a non-empty session.")
        self.agent = agent
        self.worker_tools = worker_tools
        self.store = store
        self.session = session
        self.working_directory = working_directory
        self.timeout_secs = timeout_secs
        self.max_turns = max_turns
        self.on_event = on_event

    async def dispatch(
        self,
        *,
        name: str,
        action: str,
        source_threads: Sequence[str] = (),
        timeout_secs: float | None = None,
    ) -> tuple[str, str | None]:
        store, session = self.store, self.session
        system = worker_prompt(str(self.working_directory))

        messages: list[dict[str, Any]] = []
        own = store.read(name, session=session)
        messages.extend(own_history_messages(name, own))
        for source in source_threads:
            episode = store.latest(source, session=session)
            if episode is None:
                return f"Error: source thread '{source}' has no retained episode.", None
            messages.append(source_message(episode))
        messages.append({"role": "user", "content": action})

        final = ""
        error = ""
        trace: list[dict[str, Any]] = []

        def _store(episode: Episode) -> None:
            store.append(episode)
            # ponytail: trace write is best-effort debug detail, never blocks handoff.
            with contextlib.suppress(OSError):
                store.append_trace(episode.id, trace)

        async def consume() -> None:
            nonlocal final, error
            async for event in self.agent.run(
                system=system,
                messages=messages,
                tools=list(self.worker_tools),
                max_turns=self.max_turns,
            ):
                if self.on_event is not None:
                    self.on_event(name, event)
                trace.append(_event_to_dict(event))
                if isinstance(event, AssistantEnd) and event.text.strip():
                    final = event.text.strip()
                elif isinstance(event, AgentError):
                    error = event.message

        timeout = timeout_secs or self.timeout_secs
        try:
            await asyncio.wait_for(consume(), timeout=timeout)
        except TimeoutError:
            _store(Episode(name, action, "", TIMED_OUT, session=session))
            return f"Error: thread '{name}' timed out after {int(timeout)}s.", None
        except asyncio.CancelledError:
            _store(Episode(name, action, "", CANCELLED, session=session))
            raise

        if not final:
            episode = Episode(name, action, "", ERROR, session=session)
            _store(episode)
            detail = f": {error}" if error else ""
            return f"Error: thread '{name}' produced no episode{detail}", None

        episode = Episode(name, action, final, OK, session=session)
        _store(episode)
        return final, episode.id


__all__ = ["DEFAULT_MAX_TURNS", "DEFAULT_TIMEOUT_SECS", "Dispatcher", "WorkerListener"]
