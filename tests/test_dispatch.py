"""Dispatch behaviour with a deterministic Tau provider."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from pathlib import Path

import pytest
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
from loom.threads import create_thread_tools, plan_waves

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


def _user_text(call: tuple[str, str, list[object], list[object]]) -> str:
    return " ".join(m.text for m in call[2] if isinstance(m, UserMessage))


def _episodes(result: AgentToolResult) -> list[Mapping[str, str]]:
    """The per-episode CLI metadata carried alongside the model-facing text."""
    assert isinstance(result.details, dict)
    return list(result.details["episodes"])


def test_plan_waves_orders_in_batch_dependencies() -> None:
    waves, deps = plan_waves(["a", "b", "synth"], [[], [], ["a", "b"]])
    assert waves == [[0, 1], [2]]
    assert deps == [[], [], [0, 1]]


def test_plan_waves_ignores_sources_outside_the_batch() -> None:
    waves, deps = plan_waves(["a"], [["research"]])
    assert waves == [[0]]
    assert deps == [[]]


def test_plan_waves_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="Duplicate thread name 'a'"):
        plan_waves(["a", "a"], [[], []])


def test_plan_waves_rejects_cycles() -> None:
    with pytest.raises(ValueError, match="Circular dependency"):
        plan_waves(["a", "b"], [["b"], ["a"]])
    with pytest.raises(ValueError, match="Circular dependency"):
        plan_waves(["a"], [["a"]])


def test_batch_runs_independent_items_concurrently(tmp_path: Path) -> None:
    wait = ToolCall(id="w", name="wait", arguments={})
    provider = FakeProvider(
        [
            tool_call_stream(wait),
            tool_call_stream(wait),
            text_stream("a done"),
            text_stream("b done"),
        ]
    )
    episodes = store(tmp_path)
    tools = create_thread_tools(
        provider=provider, model=MODEL, worker_tools=[wait_tool()], store=episodes
    )
    batch = next(t for t in tools if t.name == "thread_batch")

    async def scenario() -> tuple[float, str]:
        start = time.monotonic()
        result = await batch.execute(
            "1", {"items": [{"name": "a", "action": "one"}, {"name": "b", "action": "two"}]}
        )
        return time.monotonic() - start, result.text

    elapsed, text = asyncio.run(scenario())
    # Serial would be at least 2 x 0.05s of tool sleep.
    assert elapsed < 0.09
    assert "a done" in text and "b done" in text
    assert [e.content for e in episodes.read("a")] == ["a done"]
    assert [e.content for e in episodes.read("b")] == ["b done"]


def test_batch_dependent_item_sees_its_sources(tmp_path: Path) -> None:
    provider = FakeProvider(
        [text_stream("a done"), text_stream("b done"), text_stream("synth done")]
    )
    episodes = store(tmp_path)
    tools = create_thread_tools(provider=provider, model=MODEL, worker_tools=[], store=episodes)
    batch = next(t for t in tools if t.name == "thread_batch")

    async def scenario() -> AgentToolResult:
        return await batch.execute(
            "1",
            {
                "items": [
                    {"name": "a", "action": "explore a"},
                    {"name": "b", "action": "explore b"},
                    {"name": "synth", "action": "combine", "threads": ["a", "b"]},
                ]
            },
        )

    result = asyncio.run(scenario())
    assert "synth done" in result.text
    assert [e["name"] for e in _episodes(result)] == ["a", "b", "synth"]
    assert [e["text"] for e in _episodes(result)][2] == "synth done"
    injected = next(_user_text(call) for call in provider.calls if "combine" in _user_text(call))
    assert "a done" in injected and "b done" in injected
    assert [e.content for e in episodes.read("synth")] == ["synth done"]


def test_batch_skips_dependent_when_source_fails(tmp_path: Path) -> None:
    # One stream for two independent items: 'a' consumes it, 'b' answers nothing.
    provider = FakeProvider([text_stream("a done")])
    episodes = store(tmp_path)
    tools = create_thread_tools(provider=provider, model=MODEL, worker_tools=[], store=episodes)
    batch = next(t for t in tools if t.name == "thread_batch")

    async def scenario() -> AgentToolResult:
        return await batch.execute(
            "1",
            {
                "items": [
                    {"name": "a", "action": "one"},
                    {"name": "b", "action": "two"},
                    {"name": "synth", "action": "combine", "threads": ["b"]},
                ]
            },
        )

    result = asyncio.run(scenario())
    assert "source thread 'b' failed" in result.text
    assert _episodes(result)[2]["text"].startswith("Error: source thread 'b' failed")
    assert episodes.read("a") != []
    assert episodes.read("b") == []
    assert episodes.read("synth") == []


def test_batch_rejects_duplicate_names_before_dispatching(tmp_path: Path) -> None:
    provider = FakeProvider([])
    episodes = store(tmp_path)
    tools = create_thread_tools(provider=provider, model=MODEL, worker_tools=[], store=episodes)
    batch = next(t for t in tools if t.name == "thread_batch")

    async def scenario() -> str:
        result = await batch.execute(
            "1", {"items": [{"name": "a", "action": "one"}, {"name": "a", "action": "two"}]}
        )
        return result.text

    text = asyncio.run(scenario())
    assert "Duplicate thread name 'a'" in text
    assert episodes.names() == []


def test_thread_result_carries_its_single_episode(tmp_path: Path) -> None:
    provider = FakeProvider([text_stream("a done")])
    episodes = store(tmp_path)
    tools = create_thread_tools(provider=provider, model=MODEL, worker_tools=[], store=episodes)
    thread = next(t for t in tools if t.name == "thread")

    result = asyncio.run(thread.execute("1", {"name": "impl", "action": "one"}))
    assert [e["name"] for e in _episodes(result)] == ["impl"]
    assert _episodes(result)[0]["text"] == "a done"


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
