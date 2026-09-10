"""Agent loop on any-llm: sequential tool calls, OpenAI-dict messages,
tiny event set for the CLI.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolResult:
    text: str
    is_error: bool = False
    details: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    execute_fn: Any  # async (args: dict) -> ToolResult

    async def execute(self, args: Mapping[str, Any]) -> ToolResult:
        try:
            return await self.execute_fn(dict(args))
        except Exception as exc:  # tools are an isolation boundary
            return ToolResult(text=f"Error: {exc}", is_error=True)


# Events: just enough for cli.py / dispatch.py to print + collect episodes.
@dataclass(frozen=True, slots=True)
class TextDelta:
    delta: str


@dataclass(frozen=True, slots=True)
class ToolStart:
    tool_name: str
    args: dict[str, Any]
    call_id: str = ""


@dataclass(frozen=True, slots=True)
class ToolEnd:
    tool_name: str
    result: ToolResult
    is_error: bool
    call_id: str = ""


@dataclass(frozen=True, slots=True)
class AssistantEnd:
    text: str
    # Native tool calls for this turn: [{id, name, arguments}]. Empty when the
    # turn is plain text, so resume can replay the exact assistant turn.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AgentError:
    message: str


@dataclass(frozen=True, slots=True)
class RetryAttempt:
    attempt: int
    message: str


Event = TextDelta | ToolStart | ToolEnd | AssistantEnd | AgentError | RetryAttempt

MAX_CONSECUTIVE_ERRORS = 3


def to_openai_tools(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def split_model(model: str, default_provider: str | None = None) -> tuple[str, str]:
    """Accept 'provider:model' or plain model + separate provider."""
    if ":" in model:
        provider, _, model_id = model.partition(":")
        return provider or (default_provider or "openai"), model_id
    return (default_provider or "openai"), model


async def run_loop(
    *,
    provider: str | None,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: Sequence[Tool],
    max_turns: int = 32,
    api_key: str | None = None,
    api_base: str | None = None,
) -> AsyncIterator[Event]:
    """One agent run: call model, execute tool calls sequentially, repeat.

    `messages` uses OpenAI dict format. Mutated in place (assistant + tool
    turns appended) so callers can inspect history; the final assistant text
    arrives as AssistantEnd per turn, AgentError after consecutive turn
    errors (`MAX_CONSECUTIVE_ERRORS`) or max_turns.
    """
    from any_llm import AnyLLM  # lazy: keeps import cheap for tests

    provider_name, model_id = split_model(model, provider)
    api_key = api_key or os.getenv("LOOM_LLM_PROVIDER_API_KEY")
    api_base = api_base or os.getenv("LOOM_LLM_PROVIDER_BASE_URL")
    llm = AnyLLM.create(provider_name, api_key=api_key, api_base=api_base)
    by_name = {t.name: t for t in tools}
    wire_tools = to_openai_tools(tools) if tools else None

    history: list[Any] = [{"role": "system", "content": system}, *messages]

    errors = 0
    for _ in range(max(1, max_turns)):
        try:
            result = await llm.acompletion(
                model=model_id,
                messages=history,
                tools=wire_tools,
            )
            # ponytail: parse inside try — a malformed provider response is a
            # retryable turn error, never a crash that skips episode storage.
            msg = result.choices[0].message
            text: str = msg.content or ""
            raw_calls = getattr(msg, "tool_calls", None) or []
        except Exception as exc:
            errors += 1
            yield RetryAttempt(attempt=errors, message=str(exc))
            if errors >= MAX_CONSECUTIVE_ERRORS:
                yield AgentError(message=str(exc))
                return
            continue
        errors = 0

        if text:
            yield TextDelta(delta=text)

        calls: list[tuple[str, str, dict[str, Any]]] = []
        for call in raw_calls:
            fn = getattr(call, "function", None)
            if fn is None:
                continue
            try:
                args = json.loads(fn.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append((call.id, fn.name, args))

        # Replay assistant turn so the next request sees tool calls.
        history.append(
            {
                "role": "assistant",
                "content": text,
                **(
                    {
                        "tool_calls": [
                            {
                                "id": cid,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)},
                            }
                            for cid, name, args in calls
                        ]
                    }
                    if calls
                    else {}
                ),
            }
        )
        yield AssistantEnd(
            text=text,
            tool_calls=[
                {"id": cid, "name": name, "arguments": args} for cid, name, args in calls
            ],
        )

        if not calls:
            return

        for cid, name, args in calls:
            tool = by_name.get(name)
            yield ToolStart(tool_name=name, args=args, call_id=cid)
            if tool is None:
                result_ = ToolResult(text=f"Error: tool {name} not found", is_error=True)
            else:
                result_ = await tool.execute(args)
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": cid,
                    "name": name,
                    "content": result_.text,
                }
            )
            yield ToolEnd(tool_name=name, result=result_, is_error=result_.is_error, call_id=cid)

    yield AgentError(message=f"Agent stopped after max_turns={max_turns}")


__all__ = [
    "AgentError",
    "AssistantEnd",
    "Event",
    "RetryAttempt",
    "TextDelta",
    "Tool",
    "ToolEnd",
    "ToolResult",
    "ToolStart",
    "run_loop",
    "split_model",
    "to_openai_tools",
]
