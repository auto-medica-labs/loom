"""Orchestrator session transcripts: the stdout history that survives the run.

Episodes keep worker results; sessions keep the orchestrator's side. One JSONL
file per run in `<cwd>/.loom/sessions/`, one line per turn in order:

    {"type": "input", "text": "<prompt>"}
    {"type": "output", "text": "<orchestrator text>"}
    {"type": "episode", "label": "<thread>", "id": "<episode id>"}

Episode lines are references: content lives once in the episode store
(`render(..., store)` resolves them). `text` stays pure user/LLM content.

`--session <id>` prepends the rendered transcript as context and appends the
new run to the same file (`--resume` continues the latest session).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from loom.episodes import EpisodeStore

INPUT = "input"
OUTPUT = "output"
EPISODE = "episode"
ASSISTANT = "assistant"
TOOL = "tool"


class SessionStore:
    """One `<id>.jsonl` per orchestrator run, one `{type, text}` line per turn."""

    def __init__(self, path: str | Path) -> None:
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_of(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.jsonl"

    def start(self, prompt: str) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        session_id = f"{stamp}-{uuid4().hex[:6]}"
        self.log_input(session_id, prompt)
        return session_id

    def log_input(self, session_id: str, text: str) -> None:
        self._append(session_id, {"type": INPUT, "text": text})

    def log_output(self, session_id: str, text: str, label: str | None = None) -> None:
        record = {"type": OUTPUT, "text": text}
        if label:
            record["label"] = label
        self._append(session_id, record)

    def log_episode_ref(
        self, session_id: str, name: str, episode_id: str, tool_call_id: str
    ) -> None:
        # `tool_call_id` pairs the ref with its assistant turn. No legacy:
        # refs without one are skipped on replay.
        record = {"type": EPISODE, "label": name, "id": episode_id, "tool_call_id": tool_call_id}
        self._append(session_id, record)

    def log_assistant(
        self, session_id: str, text: str, tool_calls: list[dict[str, Any]] | None = None
    ) -> None:
        """Assistant turn with native tool calls ([{id, name, arguments}])."""
        record: dict[str, Any] = {"type": ASSISTANT, "text": text}
        if tool_calls:
            record["tool_calls"] = [
                {
                    "id": str(call.get("id", "")),
                    "name": str(call.get("name", "")),
                    "arguments": call.get("arguments", {}),
                }
                for call in tool_calls
                if isinstance(call, dict)
            ]
        self._append(session_id, record)

    def log_tool(self, session_id: str, label: str, text: str, tool_call_id: str) -> None:
        """Tool result inline (failed dispatches with no episode id)."""
        self._append(
            session_id,
            {"type": TOOL, "label": label, "text": text, "tool_call_id": tool_call_id},
        )

    def messages(
        self, session_id: str, store: EpisodeStore | None = None
    ) -> list[dict[str, Any]] | None:
        """Native OpenAI messages replaying the transcript, or None if unknown id.

        Roles survive the round-trip (user/assistant/tool) so --resume restores
        turn structure instead of flattening history into one user message.
        Stored text replays verbatim: no headers or labels are added.
        Episode refs always carry tool_call_id (no legacy files — delete
        .loom when upgrading); a ref without one cannot pair with its
        assistant turn and is skipped. Consecutive episode refs sharing one call
        (a thread_batch turn) merge into a single tool message, joined by a
        blank line. A trailing assistant turn with unsatisfied tool_calls
        (crashed run) is trimmed.
        """
        path = self.path_of(session_id)
        if not path.exists():
            return None
        replay: list[dict[str, Any]] = []
        pending_cid: str | None = None
        pending_parts: list[tuple[str, str]] = []

        def flush_pending() -> None:
            nonlocal pending_cid, pending_parts
            if pending_cid is not None and pending_parts:
                replay.append(
                    {
                        "role": "tool",
                        "tool_call_id": pending_cid,
                        "name": pending_parts[0][0],
                        "content": "\n\n".join(content for _, content in pending_parts),
                    }
                )
            pending_cid, pending_parts = None, []

        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            if record.get("type") == EPISODE and record.get("tool_call_id"):
                cid = str(record["tool_call_id"])
                if pending_cid is not None and cid != pending_cid:
                    flush_pending()
                pending_cid = cid
                pending_parts.append(
                    (_episode_label(record), _episode_content(record, store))
                )
                continue
            flush_pending()
            message = _record_to_message(record, store)
            if message is not None:
                replay.append(message)
        flush_pending()
        while replay and replay[-1].get("role") == "assistant" and replay[-1].get("tool_calls"):
            replay.pop()
        return replay

    def _append(self, session_id: str, record: dict) -> None:
        with self.path_of(session_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def render(self, session_id: str, store: EpisodeStore | None = None) -> str | None:
        """Full transcript as one context block, or None if unknown id.

        Pass the episode store to resolve episode references into content.
        """
        path = self.path_of(session_id)
        if not path.exists():
            return None
        turns: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            turns.append(_render_record(record, store))
        turns = [t for t in turns if t]
        return "\n\n".join([f"=== Session {session_id} ===", *turns])

    def ids(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.jsonl"))


def _render_record(record: object, store: EpisodeStore | None = None) -> str:
    if not isinstance(record, dict):
        return ""
    kind, text = record.get("type"), str(record.get("text", ""))
    label = str(record.get("label", ""))
    if kind == INPUT:
        return f"## Input\n{text}"
    if kind == EPISODE:
        episode_id = str(record.get("id", ""))
        if episode_id and store is not None:
            episode = store.get(episode_id)
            if episode is not None:
                return f"== {label} ==\n{episode.content}"
        return f"== {label} == [{episode_id}]" if episode_id else ""
    if kind == OUTPUT:
        return f">> {label}\n{text}" if label else text
    if kind == ASSISTANT:
        calls = record.get("tool_calls") or []
        names = ", ".join(str(c.get("name", "?")) for c in calls if isinstance(c, dict))
        if text and names:
            return f"{text}\n[calls: {names}]"
        return text or (f"[calls: {names}]" if names else "")
    if kind == TOOL:
        body = text or f"[{record.get('tool_call_id', '')}]"
        return f"<< {label}\n{body}" if label else body
    return ""


def _episode_label(record: dict) -> str:
    return str(record.get("label", ""))


def _episode_content(record: dict, store: EpisodeStore | None) -> str:
    episode_id = str(record.get("id", ""))
    if episode_id and store is not None:
        episode = store.get(episode_id)
        if episode is not None:
            return episode.content
    return ""


def _record_to_message(record: dict, store: EpisodeStore | None) -> dict[str, Any] | None:
    """One JSONL record to its OpenAI message; None for unknown records."""
    kind = record.get("type")
    label = str(record.get("label", ""))
    text = str(record.get("text", ""))
    if kind == INPUT:
        return {"role": "user", "content": text}
    if kind == OUTPUT:
        return {"role": "assistant", "content": text}
    if kind == ASSISTANT:
        message: dict[str, Any] = {"role": "assistant", "content": text}
        wire = []
        for call in record.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            wire.append(
                {
                    "id": str(call.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": str(call.get("name", "")),
                        "arguments": json.dumps(call.get("arguments", {})),
                    },
                }
            )
        if wire:
            message["tool_calls"] = wire
        return message
    if kind == EPISODE:
        # Unreachable via messages(): cid-carrying refs merge in the loop.
        # A ref without tool_call_id cannot pair with an assistant turn —
        # skip it (no legacy files).
        return None
    if kind == TOOL:
        return {
            "role": "tool",
            "tool_call_id": str(record.get("tool_call_id", "")),
            "name": label,
            "content": text,
        }
    return None
