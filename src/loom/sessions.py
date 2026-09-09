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
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from loom.episodes import EpisodeStore

INPUT = "input"
OUTPUT = "output"
EPISODE = "episode"


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

    def log_episode_ref(self, session_id: str, name: str, episode_id: str) -> None:
        self._append(session_id, {"type": EPISODE, "label": name, "id": episode_id})

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
    return ""
