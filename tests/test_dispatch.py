"""Dispatch behaviour with a deterministic Tau provider."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

from tau_agent.messages import AssistantMessage, TextContent, ToolCall, UserMessage
from tau_agent.provider_events import (
    AssistantDoneEvent,
    AssistantStartEvent,
    TextDeltaEvent,
    ToolCallEndEvent,
)
from tau_agent.tools import AgentTool, AgentToolResult
from tau_agent.types import JSONValue
from tau_ai import FakeProvider

from loom.dispatch import run_dispatch
from loom.episodes import EpisodeStore
from loom.threads import create_thread_tools

MODEL = "fake"


def text_stream(text: str) -> list[object]:
    """One assistant turn that answers with `text` and stops."""
    return [
        AssistantStartEvent(partial=AssistantMessage(model=MODEL)),
        TextDeltaEvent(content_index=0, delta=text, partial=AssistantMessage(content=text)),
        AssistantDoneEvent(
            reason="stop",
            message=AssistantMessage(content=[TextContent(text=text)], model=MODEL),
        ),
    ]


def store(tmp_path: Path) -> EpisodeStore:
    return EpisodeStore(tmp_path / "episodes.jsonl")


def test_dispatch_returns_and_stores_the_episode(tmp_path: Path) -> None:
    provider = FakeProvider([text_stream("wrote parser, tests pass")])
    episodes = store(tmp_path)

    answer = asyncio.run(
        run_dispatch(
            provider=provider,
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="impl",
            action="write the parser",
        )
    )

    assert answer == "wrote parser, tests pass"
    assert [e.content for e in episodes.read("impl")] == ["wrote parser, tests pass"]


def test_reused_thread_sees_its_own_history(tmp_path: Path) -> None:
    provider = FakeProvider(
        [text_stream("first pass"), text_stream("second pass")],
    )
    episodes = store(tmp_path)

    async def scenario() -> None:
        await run_dispatch(
            provider=provider,
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="impl",
            action="start",
        )
        await run_dispatch(
            provider=provider,
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="impl",
            action="continue",
        )

    asyncio.run(scenario())
    second_call_messages = provider.calls[1][2]
    history = " ".join(m.text for m in second_call_messages if isinstance(m, UserMessage))
    assert "first pass" in history
    assert "continue" in history


def test_source_thread_injects_only_its_latest_episode(tmp_path: Path) -> None:
    provider = FakeProvider(
        [text_stream("research v1"), text_stream("research v2"), text_stream("done")]
    )
    episodes = store(tmp_path)

    async def scenario() -> None:
        await run_dispatch(
            provider=provider,
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="research",
            action="look",
        )
        await run_dispatch(
            provider=provider,
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="research",
            action="look again",
        )
        await run_dispatch(
            provider=provider,
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="impl",
            action="build",
            source_threads=["research"],
        )

    asyncio.run(scenario())
    injected = " ".join(m.text for m in provider.calls[2][2] if isinstance(m, UserMessage))
    assert "research v2" in injected
    assert "research v1" not in injected


def test_missing_source_thread_is_reported(tmp_path: Path) -> None:
    provider = FakeProvider([])
    episodes = store(tmp_path)

    answer = asyncio.run(
        run_dispatch(
            provider=provider,
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="impl",
            action="build",
            source_threads=["nope"],
        )
    )

    assert "no retained episode" in answer
    assert episodes.read("impl") == []


def test_threads_tool_lists_threads_and_counts(tmp_path: Path) -> None:
    provider = FakeProvider([text_stream("a"), text_stream("b")])
    episodes = store(tmp_path)
    tools = create_thread_tools(provider=provider, model=MODEL, worker_tools=[], store=episodes)
    thread = next(t for t in tools if t.name == "thread")
    listing = next(t for t in tools if t.name == "threads")

    async def scenario() -> str:
        await thread.execute("1", {"name": "impl", "action": "one"})
        await thread.execute("2", {"name": "impl", "action": "two"})
        return (await listing.execute("3", {})).text

    assert asyncio.run(scenario()) == "Active threads:\n- impl | 2 episodes"


def tool_call_stream(call: ToolCall) -> list[object]:
    """One assistant turn that calls `call` instead of answering."""
    return [
        AssistantStartEvent(partial=AssistantMessage(model=MODEL)),
        ToolCallEndEvent(content_index=0, tool_call=call, partial=AssistantMessage(content=[call])),
        AssistantDoneEvent(
            reason="toolUse",
            message=AssistantMessage(content=[call], model=MODEL),
        ),
    ]


def wait_tool() -> AgentTool:
    async def execute(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: object = None,
        on_update: object = None,
    ) -> AgentToolResult:
        await asyncio.sleep(0.05)
        return AgentToolResult(content=[TextContent(text="waited")])

    return AgentTool(
        name="wait",
        label="wait",
        description="Sleep briefly.",
        parameters={"type": "object", "properties": {}},
        execute_fn=execute,
    )


def test_thread_tool_rejects_a_second_concurrent_dispatch(tmp_path: Path) -> None:
    call = ToolCall(id="c1", name="wait", arguments={})
    provider = FakeProvider(
        [tool_call_stream(call), text_stream("slow done"), text_stream("second")]
    )
    episodes = store(tmp_path)
    tools = create_thread_tools(
        provider=provider, model=MODEL, worker_tools=[wait_tool()], store=episodes
    )
    thread = next(t for t in tools if t.name == "thread")

    async def scenario() -> str:
        first = asyncio.create_task(thread.execute("1", {"name": "impl", "action": "one"}))
        await asyncio.sleep(0.01)
        second = await thread.execute("2", {"name": "impl", "action": "two"})
        await first
        return second.text

    assert "already running" in asyncio.run(scenario())
