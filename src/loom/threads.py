"""The orchestrator's only tools: dispatch threads and read what they left."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from tau_agent.messages import TextContent
from tau_agent.provider import ModelProvider
from tau_agent.tools import AgentTool, AgentToolResult, ToolCancellationToken, ToolUpdateCallback
from tau_agent.types import JSONValue

from loom.dispatch import DEFAULT_TIMEOUT_SECS, WorkerListener, run_dispatch
from loom.episodes import EpisodeStore, render_thread_document


def _result(text: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=text)])


def _text(arguments: Mapping[str, JSONValue], key: str) -> str:
    value = arguments.get(key)
    return str(value) if value is not None else ""


def _number(arguments: Mapping[str, JSONValue], key: str) -> float | None:
    value = arguments.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _string_list(arguments: Mapping[str, JSONValue], key: str) -> list[str]:
    value = arguments.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def create_thread_tools(
    *,
    provider: ModelProvider,
    model: str,
    worker_tools: Sequence[AgentTool],
    store: EpisodeStore,
    working_directory: str | Path = ".",
    timeout_secs: float = DEFAULT_TIMEOUT_SECS,
    on_event: WorkerListener | None = None,
) -> list[AgentTool]:
    """Tools for an orchestrator that cannot touch files itself.

    Everything it can do is: hand a bounded action to a worker, and read back
    the episode the worker stored.
    """
    active: set[str] = set()

    async def dispatch(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, on_update
        name = _text(arguments, "name")
        action = _text(arguments, "action")
        if not name or not action:
            return _result("Error: thread requires 'name' and 'action'.")
        if name in active:
            return _result(f"Error: thread '{name}' is already running; retry after it completes.")

        timeout = _number(arguments, "timeout") or timeout_secs
        active.add(name)
        try:
            episode = await run_dispatch(
                provider=provider,
                model=model,
                worker_tools=worker_tools,
                store=store,
                name=name,
                action=action,
                source_threads=_string_list(arguments, "threads"),
                working_directory=working_directory,
                timeout_secs=timeout,
                signal=signal,
                on_event=on_event,
            )
        finally:
            active.discard(name)
        return _result(episode)

    async def list_threads(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, arguments, signal, on_update
        names = store.names()
        if not names:
            return _result("No active threads in this session.")
        lines = [f"- {name} | {store.count(name)} episodes" for name in names]
        return _result("Active threads:\n" + "\n".join(lines))

    async def read_thread(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        name = _text(arguments, "name")
        if not name:
            return _result("Error: thread_read requires 'name'.")
        return _result(render_thread_document(name, store.read(name)))

    return [
        AgentTool(
            name="thread",
            label="thread",
            description=(
                "Dispatch a named worker thread. The worker reuses its own retained "
                "history and can read the latest retained episode of each named source "
                "thread. Its final response becomes the thread's next episode."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Thread name. Creates if new, reuses if existing.",
                    },
                    "action": {
                        "type": "string",
                        "description": "One bounded action for the worker.",
                    },
                    "threads": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Thread names whose latest retained episode should be loaded."
                        ),
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds for this dispatch.",
                    },
                },
                "required": ["name", "action"],
            },
            execute_fn=dispatch,
        ),
        AgentTool(
            name="threads",
            label="threads",
            description="List threads in this session and their episode counts.",
            parameters={"type": "object", "properties": {}},
            execute_fn=list_threads,
        ),
        AgentTool(
            name="thread_read",
            label="thread_read",
            description="Read the full retained episode history for one thread.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Thread name."}},
                "required": ["name"],
            },
            execute_fn=read_thread,
        ),
    ]
