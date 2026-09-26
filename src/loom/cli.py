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
    Agent,
    AgentError,
    AssistantEnd,
    RetryAttempt,
    TextDelta,
    ToolEnd,
    ToolStart,
)
from loom.dispatch import Dispatcher
from loom.engine import build_engine, credential_path, load_credentials, save_credentials
from loom.episodes import EpisodeStore
from loom.prompts import orchestrator_prompt
from loom.sessions import SessionStore
from loom.threads import Orchestrator


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


class LoomCLI:
    """One `loom "<prompt>"` run: wiring, event printing, session logging."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cwd = Path(args.cwd).resolve()
        self.store = EpisodeStore(args.store or self.cwd / ".loom" / "episodes")
        self.sessions = SessionStore(self.cwd / ".loom" / "sessions")
        self.prior_ids: list[str] = []
        self.session_id = ""
        self.pending_label = ""

    def _load_context(self) -> list[dict[str, Any]]:
        """Replay prior sessions (explicit ids and/or --resume) before the prompt."""
        messages: list[dict[str, Any]] = []
        prior_ids = list(self.args.session)
        if self.args.resume:
            ids = self.sessions.ids()
            if not ids:
                print("warning: no sessions found", file=sys.stderr)
            else:
                prior_ids.insert(0, ids[-1])
        for prior in prior_ids:
            prior_messages = self.sessions.messages(prior, self.store)
            if prior_messages is None:
                print(f"warning: session '{prior}' not found", file=sys.stderr)
            else:
                messages.extend(prior_messages)
        messages.append({"role": "user", "content": self.args.prompt})
        self.prior_ids = prior_ids
        return messages

    def _on_worker_event(self, name: str, event: object) -> None:
        if isinstance(event, ToolStart):
            print(f"    [{name}] {event.tool_name} {_preview(str(event.args))}")
        elif isinstance(event, ToolEnd):
            status = "error" if event.is_error else "ok"
            print(f"    [{name}] -> {status}: {_preview(event.result.text)}")
        elif isinstance(event, RetryAttempt):
            preview = _preview(event.message, EPISODE_PREVIEW)
            print(f"    [{name}] !! retry {event.attempt}/3: {preview}", file=sys.stderr)
        elif isinstance(event, AgentError):
            print(
                f"    [{name}] !! error: {_preview(event.message, EPISODE_PREVIEW)}",
                file=sys.stderr,
            )

    def _log_tool_result(self, result: Any, call_id: str) -> None:
        entries = _episode_entries(result)
        if entries is None:
            self.sessions.log_tool(self.session_id, self.pending_label, result.text, call_id)
            return
        for name, text, episode_id in entries:
            if episode_id:
                self.sessions.log_episode_ref(
                    self.session_id, name, episode_id, tool_call_id=call_id
                )
            else:
                self.sessions.log_tool(self.session_id, name, text, call_id)

    def _handle_event(self, event: object) -> None:
        if isinstance(event, ToolStart):
            self.pending_label = _dispatch_label(event.args)
            print(f"\n>> {self.pending_label}")
        elif isinstance(event, ToolEnd):
            self._log_tool_result(event.result, event.call_id)
            for line in _episode_lines(event.result):
                print(line)
            print()
        elif isinstance(event, AssistantEnd):
            if event.tool_calls:
                self.sessions.log_assistant(self.session_id, event.text.strip(), event.tool_calls)
            elif event.text.strip():
                self.sessions.log_output(self.session_id, event.text.strip())
        elif isinstance(event, TextDelta):
            print(event.delta, end="", flush=True)
        elif isinstance(event, RetryAttempt):
            print(
                f"!! retry {event.attempt}/3: {_preview(event.message, EPISODE_PREVIEW)}",
                file=sys.stderr,
            )
        elif isinstance(event, AgentError):
            print(f"!! error: {_preview(event.message, EPISODE_PREVIEW)}", file=sys.stderr)

    async def run(self) -> None:
        provider, model, coding = build_engine(
            provider_name=self.args.provider, model=self.args.model, cwd=self.cwd
        )
        agent = Agent(provider=provider, model=model)
        messages = self._load_context()
        self.session_id = _open_session(self.sessions, self.prior_ids, self.args.prompt)

        orchestrator = Orchestrator(
            dispatcher=Dispatcher(
                agent=agent,
                worker_tools=coding.tools(),
                store=self.store,
                session=self.session_id,
                working_directory=self.cwd,
                on_event=self._on_worker_event,
            )
        )

        async for event in agent.run(
            system=orchestrator_prompt(str(self.cwd)),
            messages=messages,
            tools=orchestrator.tools(),
            max_turns=self.args.max_turns,
        ):
            self._handle_event(event)

        print(f"\nsession: {self.sessions.path_of(self.session_id)}")


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
    asyncio.run(LoomCLI(args).run())


if __name__ == "__main__":
    main()
