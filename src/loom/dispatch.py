"""One thread dispatch: build a fresh worker context, run it, keep the episode."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

from tau_agent.events import AgentEvent, MessageEndEvent
from tau_agent.loop import run_agent_loop
from tau_agent.messages import AssistantMessage, UserMessage
from tau_agent.provider import CancellationToken, ModelProvider
from tau_agent.tools import AgentTool

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

WorkerListener = Callable[[str, AgentEvent], None]


def error_text(message: AssistantMessage) -> str:
    """Why a model turn produced nothing: provider error or diagnostic."""
    if message.error_message:
        return message.error_message
    for diagnostic in message.diagnostics or ():
        if diagnostic.error is not None and diagnostic.error.message:
            return diagnostic.error.message
        details = diagnostic.details or {}
        body = details.get("body") or details.get("message") or ""
        if body:
            return f"{diagnostic.type}: {body}"
    return ""


async def run_dispatch(
    *,
    provider: ModelProvider,
    model: str,
    worker_tools: Sequence[AgentTool],
    store: EpisodeStore,
    name: str,
    action: str,
    source_threads: Sequence[str] = (),
    working_directory: str | Path = ".",
    timeout_secs: float = DEFAULT_TIMEOUT_SECS,
    max_turns: int = DEFAULT_MAX_TURNS,
    signal: CancellationToken | None = None,
    on_event: WorkerListener | None = None,
) -> str:
    """Run one action in a worker and return the episode it handed back.

    The worker only ever sees: the worker prompt, this thread's own retained
    episodes, the latest episode of each source thread, and the action. Its
    final response is stored as the next episode for `name` and returned.
    """
    system = worker_prompt(str(working_directory))

    messages: list[UserMessage] = []
    own = store.read(name)
    if own:
        messages.append(UserMessage(content=render_self_context(name, own)))
    for source in source_threads:
        episode = store.latest(source)
        if episode is None:
            return f"Error: source thread '{source}' has no retained episode."
        messages.append(UserMessage(content=render_source_context(episode)))
    messages.append(UserMessage(content=action))

    final = ""
    error = ""

    async def consume() -> None:
        nonlocal final, error
        async for event in run_agent_loop(
            provider=provider,
            model=model,
            system=system,
            messages=list(messages),
            tools=list(worker_tools),
            max_turns=max_turns,
            signal=signal,
        ):
            if on_event is not None:
                on_event(name, event)
            if isinstance(event, MessageEndEvent) and isinstance(event.message, AssistantMessage):
                text = event.message.text.strip()
                if text:
                    final = text
                failure = error_text(event.message)
                if failure:
                    error = failure

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
