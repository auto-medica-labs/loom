"""The orchestrator's only tools: dispatch threads and read what they left."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from loom.agent import Tool, ToolResult
from loom.dispatch import Dispatcher
from loom.episodes import EpisodeStore, render_thread_document


def _result(text: str) -> ToolResult:
    return ToolResult(text=text, is_error=text.startswith("Error:"))


def _with_episodes(text: str, episodes: Sequence[tuple[str, str, str | None]]) -> ToolResult:
    return ToolResult(
        text=text,
        is_error=False,
        details={
            "episodes": [
                {"name": name, "text": body, "id": episode_id}
                for name, body, episode_id in episodes
            ]
        },
    )


def _stored_id(store: EpisodeStore, name: str, text: str, session: str) -> str | None:
    """Id of the episode just stored, or None for failed dispatches."""
    if text.startswith("Error:"):
        return None
    latest = store.latest(name, session=session)
    return latest.id if latest is not None and latest.content == text else None


def _text(arguments: Mapping[Any, Any], key: str) -> str:
    value = arguments.get(key)
    return str(value) if value is not None else ""


def _number(arguments: Mapping[Any, Any], key: str) -> float | None:
    value = arguments.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _string_list(arguments: Mapping[Any, Any], key: str) -> list[str]:
    value = arguments.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def plan_waves(
    names: Sequence[str], sources: Sequence[Sequence[str]]
) -> tuple[list[list[int]], list[list[int]]]:
    """Topological waves for one batch, plus each item's in-batch dependencies.

    Only sources that name another thread in the same batch become edges: sources
    from earlier turns are already in the store and need no ordering. Raises
    `ValueError` on a duplicate name or a cycle, in which case no item may run.
    """
    position_of: dict[str, int] = {}
    for position, name in enumerate(names):
        if name in position_of:
            raise ValueError(f"Duplicate thread name '{name}' in parallel dispatch.")
        position_of[name] = position

    deps: list[list[int]] = [[] for _ in names]
    dependents: list[list[int]] = [[] for _ in names]
    pending = [0] * len(names)
    for position, source_names in enumerate(sources):
        for source in dict.fromkeys(source_names):
            if source not in position_of:
                continue
            if position_of[source] == position:
                raise ValueError(
                    f"Circular dependency in thread dispatch: "
                    f"'{names[position]}' depends on itself."
                )
            deps[position].append(position_of[source])
            dependents[position_of[source]].append(position)
            pending[position] += 1

    waves: list[list[int]] = []
    done = [False] * len(names)
    remaining = len(names)
    while remaining:
        wave = [i for i in range(len(names)) if not done[i] and not pending[i]]
        if not wave:
            stuck = ", ".join(names[i] for i in range(len(names)) if not done[i])
            raise ValueError(f"Circular dependency in thread dispatch: {stuck}.")
        for position in wave:
            done[position] = True
            remaining -= 1
            for dependent in dependents[position]:
                pending[dependent] -= 1
        waves.append(wave)
    return waves, deps


class Orchestrator:
    """Tools for an orchestrator that cannot touch files itself.

    Everything it can do is: hand a bounded action to a worker, and read back
    the episode the worker stored.
    """

    def __init__(self, *, dispatcher: Dispatcher) -> None:
        self.dispatcher = dispatcher
        self.store = dispatcher.store
        self.session = dispatcher.session
        self._active: set[str] = set()

    async def dispatch_thread(self, args: dict[str, Any]) -> ToolResult:
        name = _text(args, "name")
        action = _text(args, "action")
        if not name or not action:
            return _result("Error: thread requires 'name' and 'action'.")
        if name in self._active:
            return _result(f"Error: thread '{name}' is already running; retry after it completes.")

        self._active.add(name)
        try:
            episode = await self.dispatcher.dispatch(
                name=name,
                action=action,
                source_threads=_string_list(args, "threads"),
                timeout_secs=_number(args, "timeout"),
            )
        finally:
            self._active.discard(name)
        return _with_episodes(
            episode, [(name, episode, _stored_id(self.store, name, episode, self.session))]
        )

    async def dispatch_batch(self, args: dict[str, Any]) -> ToolResult:
        items = args.get("items")
        if not isinstance(items, list) or not items:
            return _result("Error: thread_batch requires a non-empty 'items' list.")

        names: list[str] = []
        actions: list[str] = []
        sources: list[list[str]] = []
        timeouts: list[float | None] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                return _result(f"Error: thread_batch item {index} must be an object.")
            name, action = _text(item, "name"), _text(item, "action")
            if not name or not action:
                return _result(f"Error: thread_batch item {index} requires 'name' and 'action'.")
            names.append(name)
            actions.append(action)
            sources.append(_string_list(item, "threads"))
            timeouts.append(_number(item, "timeout"))

        try:
            waves, deps = plan_waves(names, sources)
        except ValueError as error:
            return _result(f"Error: {error}")

        busy = sorted(set(names) & self._active)
        if busy:
            return _result(
                f"Error: thread(s) {', '.join(busy)} already running; retry after they complete."
            )

        results: dict[int, str] = {}
        failed: set[int] = set()
        self._active.update(names)
        try:
            for wave in waves:
                runnable: list[int] = []
                for index in wave:
                    dead = next((names[d] for d in deps[index] if d in failed), None)
                    if dead is not None:
                        results[index] = (
                            f"Error: source thread '{dead}' failed; "
                            f"dispatch '{names[index]}' skipped."
                        )
                        failed.add(index)
                    else:
                        runnable.append(index)
                if not runnable:
                    continue
                outcomes = await asyncio.gather(
                    *(
                        self.dispatcher.dispatch(
                            name=names[index],
                            action=actions[index],
                            source_threads=sources[index],
                            timeout_secs=timeouts[index],
                        )
                        for index in runnable
                    ),
                    return_exceptions=True,
                )
                for index, outcome in zip(runnable, outcomes, strict=True):
                    if isinstance(outcome, BaseException):
                        results[index] = f"Error: thread '{names[index]}' failed: {outcome}"
                        failed.add(index)
                    else:
                        results[index] = outcome
                        if outcome.startswith("Error:"):
                            failed.add(index)
        finally:
            self._active.difference_update(names)

        body = "\n".join(f"== {names[i]} ==\n{results[i]}" for i in range(len(names)))
        entries = [
            (names[i], results[i], _stored_id(self.store, names[i], results[i], self.session))
            for i in range(len(names))
        ]
        return _with_episodes(body, entries)

    async def list_threads(self, args: dict[str, Any]) -> ToolResult:
        names = self.store.names(session=self.session)
        if not names:
            return _result("No active threads in this session.")
        lines = [
            f"- {name} | {self.store.count(name, session=self.session)} episodes" for name in names
        ]
        return _result("Active threads:\n" + "\n".join(lines))

    async def read_thread(self, args: dict[str, Any]) -> ToolResult:
        name = _text(args, "name")
        if not name:
            return _result("Error: thread_read requires 'name'.")
        return _result(render_thread_document(name, self.store.read(name, session=self.session)))

    def tools(self) -> list[Tool]:
        return [
            Tool(
                name="thread",
                description=(
                    "Dispatch a named worker thread. The worker reuses its own retained "
                    "history and can read the latest retained episode of each named source "
                    "thread. Its final response becomes the thread's next episode."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "action": {"type": "string"},
                        "threads": {"type": "array", "items": {"type": "string"}},
                        "timeout": {"type": "number"},
                    },
                    "required": ["name", "action"],
                },
                execute_fn=self.dispatch_thread,
            ),
            Tool(
                name="thread_batch",
                description=(
                    "Dispatch several threads as one batch. Items with no dependency on "
                    "another item in this batch run concurrently; an item that names "
                    "another batch item as a source waits for it and receives its episode."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "action": {"type": "string"},
                                    "threads": {"type": "array", "items": {"type": "string"}},
                                    "timeout": {"type": "number"},
                                },
                                "required": ["name", "action"],
                            },
                        }
                    },
                    "required": ["items"],
                },
                execute_fn=self.dispatch_batch,
            ),
            Tool(
                name="threads",
                description="List threads in this session and their episode counts.",
                parameters={"type": "object", "properties": {}},
                execute_fn=self.list_threads,
            ),
            Tool(
                name="thread_read",
                description="Read the full retained episode history for one thread.",
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                execute_fn=self.read_thread,
            ),
        ]


__all__ = ["Orchestrator", "plan_waves"]
