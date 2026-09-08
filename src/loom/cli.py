"""Print-mode orchestrator: `loom "refactor the parser"`.

The orchestrator only holds thread tools; workers get Tau's coding tools.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from tau_agent.events import (
    MessageEndEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from tau_agent.loop import run_agent_loop
from tau_agent.messages import AssistantMessage, UserMessage
from tau_agent.provider_events import TextDeltaEvent
from tau_agent.tools import AgentTool, AgentToolResult
from tau_agent.types import JSONValue

from loom.dispatch import error_text
from loom.engine import build_engine
from loom.episodes import EpisodeStore
from loom.prompts import orchestrator_prompt
from loom.threads import create_thread_tools


def _get_version() -> str:
    try:
        return f"loom {version('loom')}"
    except PackageNotFoundError:
        return "loom unknown"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="loom", description="Thread-and-episode orchestration on Tau."
    )
    parser.add_argument("prompt", help="What to work on.")
    parser.add_argument("--provider", default=None, help="Tau provider name.")
    parser.add_argument("--model", default=None, help="Model id.")
    parser.add_argument("--cwd", default=".", help="Working directory for workers.")
    parser.add_argument(
        "--store",
        default=None,
        help="Episode file (default: <cwd>/.loom/episodes.jsonl).",
    )
    parser.add_argument("--max-turns", type=int, default=32, help="Orchestrator turns.")
    parser.add_argument(
        "--version",
        action="version",
        version=_get_version(),
        help="Show the loom version and exit.",
    )
    return parser.parse_args(argv)


def _preview(text: str, limit: int = 220) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


EPISODE_PREVIEW = 300


def _episode_lines(result: AgentToolResult) -> list[str]:
    """One line per episode, whichever tool produced them."""
    details = result.details
    episodes = details.get("episodes") if isinstance(details, dict) else None
    if not isinstance(episodes, list):
        return [f"<< {_preview(result.text, EPISODE_PREVIEW)}"]
    lines = []
    for episode in episodes:
        if not isinstance(episode, Mapping):
            continue
        name = str(episode.get("name", "?"))
        text = str(episode.get("text", ""))
        marker = "!! " if text.startswith("Error:") else ""
        lines.append(
            f"<< {marker}{name} ({len(text):,} chars): {_preview(text, EPISODE_PREVIEW)}"
        )
    return lines


def _dispatch_label(arguments: Mapping[str, JSONValue]) -> str:
    items = arguments.get("items")
    if isinstance(items, list):
        names = [
            str(item["name"])
            for item in items
            if isinstance(item, Mapping) and item.get("name")
        ]
        return "batch: " + ", ".join(names)
    return f"thread {_preview(str(arguments.get('name', '')), 40)}"


async def _run(args: argparse.Namespace) -> None:
    cwd = Path(args.cwd).resolve()
    store = EpisodeStore(args.store or cwd / ".loom" / "episodes.jsonl")
    provider, model, worker_tools = build_engine(
        provider_name=args.provider, model=args.model, cwd=cwd
    )

    def on_worker_event(name: str, event: object) -> None:
        if isinstance(event, ToolExecutionStartEvent):
            print(f"    [{name}] {event.tool_name} {_preview(str(event.args))}")
        elif isinstance(event, ToolExecutionEndEvent):
            status = "error" if event.is_error else "ok"
            print(f"    [{name}] -> {status}: {_preview(event.result.text)}")

    tools: list[AgentTool] = create_thread_tools(
        provider=provider,
        model=model,
        worker_tools=worker_tools,
        store=store,
        working_directory=cwd,
        on_event=on_worker_event,
    )

    async for event in run_agent_loop(
        provider=provider,
        model=model,
        system=orchestrator_prompt(str(cwd)),
        messages=[UserMessage(content=args.prompt)],
        tools=tools,
        max_turns=args.max_turns,
        session_id=uuid.uuid4().hex,
    ):
        if isinstance(event, ToolExecutionStartEvent):
            print(f"\n>> {_dispatch_label(event.args)}")
        elif isinstance(event, ToolExecutionEndEvent):
            for line in _episode_lines(event.result):
                print(line)
            print()
        elif isinstance(event, MessageUpdateEvent) and isinstance(
            event.assistant_message_event, TextDeltaEvent
        ):
            print(event.assistant_message_event.delta, end="", flush=True)
        elif isinstance(event, MessageEndEvent) and isinstance(event.message, AssistantMessage):
            if event.message.text.strip():
                print()
            failure = error_text(event.message)
            if failure:
                print(f"error: {failure}", file=sys.stderr)

    print(f"\nepisodes: {store.path}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
