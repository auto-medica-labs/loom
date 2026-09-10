"""Dispatch behaviour with a deterministic any-llm fake."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import any_llm
import pytest

from loom.agent import Tool, ToolResult
from loom.dispatch import run_dispatch
from loom.episodes import EpisodeStore
from loom.threads import create_thread_tools, plan_waves

MODEL = "fake:model"
SID = "test-session"


class FakeLLM:
    """Scripted any-llm client. `script` items are text or tool-call specs."""

    def __init__(self, script: list[Any]):
        self._script = list(script)
        self.calls: list[list[dict[str, Any]]] = []
        self._lock = asyncio.Lock()

    async def acompletion(
        self, *, model: str, messages: list[dict[str, Any]], tools: Any = None
    ) -> Any:
        async with self._lock:
            self.calls.append([dict(m) for m in messages])
            item = self._script.pop(0) if self._script else ""
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, dict) and item.get("empty_choices"):
            return SimpleNamespace(choices=[])
        if isinstance(item, dict) and "tool_calls" in item:
            calls = [
                SimpleNamespace(
                    id=c["id"],
                    function=SimpleNamespace(
                        name=c["name"], arguments=json.dumps(c.get("args", {}))
                    ),
                )
                for c in item["tool_calls"]
            ]
            msg = SimpleNamespace(content=item.get("text", ""), tool_calls=calls)
        else:
            msg = SimpleNamespace(content=item, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


@pytest.fixture
def patch_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    fake = FakeLLM([])
    monkeypatch.setattr(any_llm.AnyLLM, "create", lambda *a, **k: fake)
    return fake


def use_script(patch_llm: FakeLLM, script: list[Any]) -> FakeLLM:
    patch_llm._script = list(script)
    return patch_llm


def store(tmp_path: Path) -> EpisodeStore:
    return EpisodeStore(tmp_path / "episodes")


def user_text(messages: list[dict[str, Any]]) -> str:
    return " ".join(m.get("content", "") for m in messages if m.get("role") == "user")


def test_dispatch_returns_and_stores_the_episode(tmp_path: Path, patch_llm: FakeLLM) -> None:
    use_script(patch_llm, ["wrote parser, tests pass"])
    episodes = store(tmp_path)

    answer = asyncio.run(
        run_dispatch(
            provider="fake",
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="impl",
            action="write the parser",
            session=SID,
        )
    )

    assert answer == "wrote parser, tests pass"
    assert [e.content for e in episodes.read("impl", SID)] == ["wrote parser, tests pass"]


def test_dispatch_stamps_the_session_id(tmp_path: Path, patch_llm: FakeLLM) -> None:
    use_script(patch_llm, ["done"])
    episodes = store(tmp_path)

    asyncio.run(
        run_dispatch(
            provider="fake",
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="impl",
            action="build",
            session="20260909-000000-abc123",
        )
    )

    assert episodes.read("impl", "20260909-000000-abc123")[0].session == "20260909-000000-abc123"
    assert [e.thread for e in episodes.by_session("20260909-000000-abc123")] == ["impl"]
    assert episodes.by_session("other") == []


def test_reused_thread_sees_its_own_history(tmp_path: Path, patch_llm: FakeLLM) -> None:
    use_script(patch_llm, ["first pass", "second pass"])
    episodes = store(tmp_path)

    async def scenario() -> None:
        await run_dispatch(
            provider="fake",
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="impl",
            action="start",
            session=SID,
        )
        await run_dispatch(
            provider="fake",
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="impl",
            action="continue",
            session=SID,
        )

    asyncio.run(scenario())
    history = user_text(patch_llm.calls[1])
    assert "first pass" in history
    assert "continue" in history


def test_source_thread_injects_only_its_latest_episode(tmp_path: Path, patch_llm: FakeLLM) -> None:
    use_script(patch_llm, ["research v1", "research v2", "done"])
    episodes = store(tmp_path)

    async def scenario() -> None:
        await run_dispatch(
            provider="fake",
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="research",
            action="look",
            session=SID,
        )
        await run_dispatch(
            provider="fake",
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="research",
            action="look again",
            session=SID,
        )
        await run_dispatch(
            provider="fake",
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="impl",
            action="build",
            source_threads=["research"],
            session=SID,
        )

    asyncio.run(scenario())
    injected = user_text(patch_llm.calls[2])
    assert "research v2" in injected
    assert "research v1" not in injected


def test_missing_source_thread_is_reported(tmp_path: Path, patch_llm: FakeLLM) -> None:
    episodes = store(tmp_path)
    answer = asyncio.run(
        run_dispatch(
            provider="fake",
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="impl",
            action="build",
            source_threads=["nope"],
            session=SID,
        )
    )
    assert "no retained episode" in answer
    assert episodes.read("impl", SID) == []


def _episodes(result: ToolResult) -> list[dict[str, str]]:
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


def wait_tool() -> Tool:
    async def execute(args: dict[str, Any]) -> ToolResult:
        await asyncio.sleep(0.05)
        return ToolResult(text="waited")

    return Tool(
        name="wait",
        description="Sleep briefly.",
        parameters={"type": "object", "properties": {}},
        execute_fn=execute,
    )


def test_batch_runs_independent_items_concurrently(tmp_path: Path, patch_llm: FakeLLM) -> None:
    use_script(
        patch_llm,
        [
            {"tool_calls": [{"id": "w1", "name": "wait"}]},
            {"tool_calls": [{"id": "w2", "name": "wait"}]},
            "a done",
            "b done",
        ],
    )
    episodes = store(tmp_path)
    tools = create_thread_tools(
        provider="fake", model=MODEL, worker_tools=[wait_tool()], store=episodes, session=SID
    )
    batch = next(t for t in tools if t.name == "thread_batch")

    async def scenario() -> tuple[float, str]:
        start = time.monotonic()
        result = await batch.execute(
            {"items": [{"name": "a", "action": "one"}, {"name": "b", "action": "two"}]}
        )
        return time.monotonic() - start, result.text

    elapsed, text = asyncio.run(scenario())
    assert elapsed < 0.09
    assert "a done" in text and "b done" in text
    assert [e.content for e in episodes.read("a", SID)] == ["a done"]
    assert [e.content for e in episodes.read("b", SID)] == ["b done"]


def test_batch_dependent_item_sees_its_sources(tmp_path: Path, patch_llm: FakeLLM) -> None:
    use_script(patch_llm, ["a done", "b done", "synth done"])
    episodes = store(tmp_path)
    tools = create_thread_tools(
        provider="fake", model=MODEL, worker_tools=[], store=episodes, session=SID
    )
    batch = next(t for t in tools if t.name == "thread_batch")

    async def scenario() -> ToolResult:
        return await batch.execute(
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
    injected = next(user_text(c) for c in patch_llm.calls if "combine" in user_text(c))
    assert "a done" in injected and "b done" in injected
    assert [e.content for e in episodes.read("synth", SID)] == ["synth done"]


def test_batch_skips_dependent_when_source_fails(tmp_path: Path, patch_llm: FakeLLM) -> None:
    use_script(patch_llm, ["a done"])
    episodes = store(tmp_path)
    tools = create_thread_tools(
        provider="fake", model=MODEL, worker_tools=[], store=episodes, session=SID
    )
    batch = next(t for t in tools if t.name == "thread_batch")

    async def scenario() -> ToolResult:
        return await batch.execute(
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
    assert episodes.read("a", SID) != []
    assert episodes.read("b", SID) == []
    assert episodes.read("synth", SID) == []


def test_batch_rejects_duplicate_names_before_dispatching(
    tmp_path: Path, patch_llm: FakeLLM
) -> None:
    episodes = store(tmp_path)
    tools = create_thread_tools(
        provider="fake", model=MODEL, worker_tools=[], store=episodes, session=SID
    )
    batch = next(t for t in tools if t.name == "thread_batch")
    text = asyncio.run(
        batch.execute({"items": [{"name": "a", "action": "one"}, {"name": "a", "action": "two"}]})
    ).text
    assert "Duplicate thread name 'a'" in text
    assert episodes.names(SID) == []


def test_thread_result_carries_the_stored_episode_id(tmp_path: Path, patch_llm: FakeLLM) -> None:
    use_script(patch_llm, ["a done"])
    episodes = store(tmp_path)
    tools = create_thread_tools(
        provider="fake", model=MODEL, worker_tools=[], store=episodes, session=SID
    )
    thread = next(t for t in tools if t.name == "thread")
    result = asyncio.run(thread.execute({"name": "impl", "action": "one"}))
    stored = episodes.read("impl", SID)[0]
    assert _episodes(result)[0]["id"] == stored.id


def test_thread_result_carries_its_single_episode(tmp_path: Path, patch_llm: FakeLLM) -> None:
    use_script(patch_llm, ["a done"])
    episodes = store(tmp_path)
    tools = create_thread_tools(
        provider="fake", model=MODEL, worker_tools=[], store=episodes, session=SID
    )
    thread = next(t for t in tools if t.name == "thread")
    result = asyncio.run(thread.execute({"name": "impl", "action": "one"}))
    assert [e["name"] for e in _episodes(result)] == ["impl"]
    assert _episodes(result)[0]["text"] == "a done"


def test_threads_tool_lists_threads_and_counts(tmp_path: Path, patch_llm: FakeLLM) -> None:
    use_script(patch_llm, ["a", "b"])
    episodes = store(tmp_path)
    tools = create_thread_tools(
        provider="fake", model=MODEL, worker_tools=[], store=episodes, session=SID
    )
    thread = next(t for t in tools if t.name == "thread")
    listing = next(t for t in tools if t.name == "threads")

    async def scenario() -> str:
        await thread.execute({"name": "impl", "action": "one"})
        await thread.execute({"name": "impl", "action": "two"})
        return (await listing.execute({})).text

    assert asyncio.run(scenario()) == "Active threads:\n- impl | 2 episodes"


def test_thread_tool_rejects_a_second_concurrent_dispatch(
    tmp_path: Path, patch_llm: FakeLLM
) -> None:
    use_script(
        patch_llm,
        [
            {"tool_calls": [{"id": "c1", "name": "wait"}]},
            "slow done",
            "second",
        ],
    )
    episodes = store(tmp_path)
    tools = create_thread_tools(
        provider="fake", model=MODEL, worker_tools=[wait_tool()], store=episodes, session=SID
    )
    thread = next(t for t in tools if t.name == "thread")

    async def scenario() -> str:
        first = asyncio.create_task(thread.execute({"name": "impl", "action": "one"}))
        await asyncio.sleep(0.01)
        second = await thread.execute({"name": "impl", "action": "two"})
        await first
        return second.text

    assert "already running" in asyncio.run(scenario())


async def _collect_loop(patch_llm: FakeLLM) -> list:
    from loom.agent import run_loop

    return [
        event
        async for event in run_loop(
            provider="fake", model=MODEL, system="sys", messages=[], tools=[]
        )
    ]


def test_run_loop_retries_transient_failures_then_recovers(patch_llm: FakeLLM) -> None:
    from loom.agent import AgentError, AssistantEnd

    use_script(patch_llm, [RuntimeError("flake"), RuntimeError("flake"), "recovered"])
    events = asyncio.run(_collect_loop(patch_llm))
    assert not [e for e in events if isinstance(e, AgentError)]
    assert [e.text for e in events if isinstance(e, AssistantEnd)] == ["recovered"]
    assert len(patch_llm.calls) == 3


def test_run_loop_gives_up_after_consecutive_errors(patch_llm: FakeLLM) -> None:
    from loom.agent import AgentError

    use_script(patch_llm, [RuntimeError("down")] * 5)
    events = asyncio.run(_collect_loop(patch_llm))
    assert any(isinstance(e, AgentError) for e in events)
    assert len(patch_llm.calls) == 3


def test_malformed_response_stores_error_episode_not_crash(
    tmp_path: Path, patch_llm: FakeLLM
) -> None:
    use_script(patch_llm, [{"empty_choices": True}] * 5)
    episodes = store(tmp_path)
    answer = asyncio.run(
        run_dispatch(
            provider="fake",
            model=MODEL,
            worker_tools=[],
            store=episodes,
            name="impl",
            action="build",
            session=SID,
        )
    )
    assert answer.startswith("Error: thread 'impl' produced no episode")
    persisted = episodes.by_session(SID)
    assert len(persisted) == 1 and persisted[0].status == "error"
    assert episodes.read_trace(persisted[0].id) != []
