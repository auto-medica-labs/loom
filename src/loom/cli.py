"""Print-mode orchestrator: `loom "refactor the parser"`.

The orchestrator only holds thread tools; workers get Tau's coding tools.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
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
from tau_agent.tools import AgentTool

from loom.engine import build_engine
from loom.episodes import EpisodeStore
from loom.prompts import orchestrator_prompt
from loom.threads import create_thread_tools


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
    return parser.parse_args(argv)


def _preview(text: str, limit: int = 220) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


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
            print(f"\n>> thread {_preview(str(event.args.get('name', '')), 40)}")
        elif isinstance(event, ToolExecutionEndEvent):
            print(f"<< {_preview(event.result.text, 400)}\n")
        elif isinstance(event, MessageUpdateEvent) and isinstance(
            event.assistant_message_event, TextDeltaEvent
        ):
            print(event.assistant_message_event.delta, end="", flush=True)
        elif (
            isinstance(event, MessageEndEvent)
            and isinstance(event.message, AssistantMessage)
            and event.message.text.strip()
        ):
            print()

    print(f"\nepisodes: {store.path}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
