"""Print-mode orchestrator: `loom "refactor the parser"`.

The orchestrator only holds thread tools; workers get coding tools.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from loom.agent import (
    AgentError,
    AssistantEnd,
    RetryAttempt,
    TextDelta,
    Tool,
    ToolEnd,
    ToolStart,
    run_loop,
)
from loom.engine import build_engine, credential_path, load_credentials, save_credentials
from loom.episodes import EpisodeStore
from loom.prompts import orchestrator_prompt
from loom.sessions import SessionStore
from loom.threads import create_thread_tools


def _get_version() -> str:
    for dist in ("loom-threads", "loom"):
        try:
            return f"loom {version(dist)}"
        except PackageNotFoundError:
            continue
    return "loom unknown"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="loom", description="Thread-and-episode orchestration (any-llm engine)."
    )
    parser.add_argument("prompt", help="What to work on.")
    parser.add_argument("--provider", default=None, help="any-llm provider name.")
    parser.add_argument("--model", default=None, help="Model id ('provider:model' or plain).")
    parser.add_argument("--cwd", default=".", help="Working directory for workers.")
    parser.add_argument(
        "--store",
        default=None,
        help="Episode directory (default: <cwd>/.loom/episodes).",
    )
    parser.add_argument("--max-turns", type=int, default=32, help="Orchestrator turns.")
    parser.add_argument(
        "--session",
        action="append",
        default=[],
        metavar="SESSION_ID",
        help="Resume a session (one ID appends in place; several start a new run).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the latest session in place.",
    )
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


def _episode_lines(result: Any) -> list[str]:
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
        lines.append(f"<< {marker}{name} ({len(text):,} chars): {_preview(text, EPISODE_PREVIEW)}")
    return lines


def _dispatch_label(arguments: Mapping[str, Any]) -> str:
    items = arguments.get("items")
    if isinstance(items, list):
        names = [
            str(item["name"]) for item in items if isinstance(item, Mapping) and item.get("name")
        ]
        return "batch: " + ", ".join(names)
    return f"thread {_preview(str(arguments.get('name', '')), 40)}"


def _open_session(sessions: SessionStore, resume: list[str], prompt: str) -> str:
    """One existing --session id continues in place; otherwise start a new run."""
    if len(resume) == 1 and sessions.path_of(resume[0]).exists():
        sessions.log_input(resume[0], prompt)
        return resume[0]
    return sessions.start(prompt)


def _episode_entries(result: Any) -> list[tuple[str, str, str | None]] | None:
    """Per-episode (name, text, id) triples in dispatch order, or None."""
    details = result.details
    episodes = details.get("episodes") if isinstance(details, dict) else None
    if not isinstance(episodes, list):
        return None
    entries = []
    for episode in episodes:
        if not isinstance(episode, Mapping):
            continue
        raw_id = episode.get("id")
        entries.append(
            (
                str(episode.get("name", "?")),
                str(episode.get("text", "")),
                raw_id if isinstance(raw_id, str) else None,
            )
        )
    return entries


async def _run(args: argparse.Namespace) -> None:
    cwd = Path(args.cwd).resolve()
    store = EpisodeStore(args.store or cwd / ".loom" / "episodes")
    provider, model, worker_tools = build_engine(
        provider_name=args.provider, model=args.model, cwd=cwd
    )
    sessions = SessionStore(cwd / ".loom" / "sessions")

    messages: list[dict[str, Any]] = []
    prior_ids = list(args.session)
    if args.resume:
        ids = sessions.ids()
        if not ids:
            print("warning: no sessions found", file=sys.stderr)
        else:
            prior_ids.insert(0, ids[-1])
    for prior in prior_ids:
        prior_messages = sessions.messages(prior, store)
        if prior_messages is None:
            print(f"warning: session '{prior}' not found", file=sys.stderr)
        else:
            messages.extend(prior_messages)
    messages.append({"role": "user", "content": args.prompt})
    session_id = _open_session(sessions, prior_ids, args.prompt)

    def on_worker_event(name: str, event: object) -> None:
        if isinstance(event, ToolStart):
            print(f"    [{name}] {event.tool_name} {_preview(str(event.args))}")
        elif isinstance(event, ToolEnd):
            status = "error" if event.is_error else "ok"
            print(f"    [{name}] -> {status}: {_preview(event.result.text)}")
        elif isinstance(event, RetryAttempt):
            print(f"    [{name}] retry {event.attempt}/3: {event.message}", file=sys.stderr)

    tools: list[Tool] = create_thread_tools(
        provider=provider,
        model=model,
        worker_tools=worker_tools,
        store=store,
        working_directory=cwd,
        on_event=on_worker_event,
        session=session_id,
    )

    pending_label = ""
    async for event in run_loop(
        provider=provider,
        model=model,
        system=orchestrator_prompt(str(cwd)),
        messages=messages,
        tools=tools,
        max_turns=args.max_turns,
    ):
        if isinstance(event, ToolStart):
            pending_label = _dispatch_label(event.args)
            print(f"\n>> {pending_label}")
        elif isinstance(event, ToolEnd):
            entries = _episode_entries(event.result)
            if entries is None:
                sessions.log_tool(session_id, pending_label, event.result.text, event.call_id)
            else:
                for name, text, episode_id in entries:
                    if episode_id:
                        sessions.log_episode_ref(
                            session_id, name, episode_id, tool_call_id=event.call_id
                        )
                    else:
                        sessions.log_tool(session_id, name, text, event.call_id)
            for line in _episode_lines(event.result):
                print(line)
            print()
        elif isinstance(event, AssistantEnd):
            if event.tool_calls:
                sessions.log_assistant(session_id, event.text.strip(), event.tool_calls)
            elif event.text.strip():
                sessions.log_output(session_id, event.text.strip())
        elif isinstance(event, TextDelta):
            print(event.delta, end="", flush=True)
        elif isinstance(event, RetryAttempt):
            print(f"retry {event.attempt}/3: {event.message}", file=sys.stderr)
        elif isinstance(event, AgentError):
            print(f"error: {event.message}", file=sys.stderr)

    print(f"\nsession: {sessions.path_of(session_id)}")


def _redact(value: str) -> str:
    return f"****{value[-4:]}" if len(value) > 4 else "****" if value else "(not set)"


def _run_setup() -> None:
    import getpass

    current = load_credentials()
    print(f"loom setup (saves to {credential_path()}, mode 600)\n")

    def ask(label: str, key: str, *, secret: bool = False) -> str:
        existing = current.get(key, "")
        hint = _redact(existing) if secret else existing or "(not set)"
        prompt = f"{label} [{hint}]: "
        value = (getpass.getpass(prompt) if secret else input(prompt)).strip()
        return value or existing

    try:
        data = {
            "base_url": ask("Provider base URL", "base_url"),
            "api_key": ask("Provider API key", "api_key", secret=True),
            "model": ask("Model (provider:model)", "model"),
            "provider": ask("Provider override (optional)", "provider"),
        }
    except (EOFError, KeyboardInterrupt):
        print("\nsetup cancelled.")
        return
    path = save_credentials({k: v for k, v in data.items() if v})
    print(f"saved to {path}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "setup":
        _run_setup()
        return
    args = _parse_args(argv)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
