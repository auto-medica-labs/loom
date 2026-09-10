"""Durable episode storage: the only thing a thread leaves behind."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

OK = "ok"
ERROR = "error"
TIMED_OUT = "timed_out"
CANCELLED = "cancelled"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _new_id() -> str:
    return uuid4().hex[:12]


@dataclass(frozen=True, slots=True)
class Episode:
    """One completed dispatch of one thread.

    `content` is the worker's final response, verbatim. `status` is `ok` for a
    handoff; anything else records a dispatch that died before answering.
    `id` names the file (`<id>.jsonl`) inside the episode directory.
    """

    thread: str
    action: str
    content: str
    status: str = OK
    created_at: str = field(default_factory=_now)
    id: str = field(default_factory=_new_id)
    session: str = field(kw_only=True)

    def __post_init__(self) -> None:
        if not self.session:
            raise ValueError("Episode requires a non-empty session.")


class EpisodeStore:
    """One file per episode in a directory (`<dir>/<id>.jsonl`).

    Episodes are write-once, read-many, and only queried by thread name.
    """

    def __init__(self, path: str | Path) -> None:
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._last_ns = 0

    def append(self, episode: Episode) -> None:
        target = self.dir / f"{episode.id}.jsonl"
        counter = 1
        while target.exists():
            target = self.dir / f"{episode.id}-{counter}.jsonl"
            counter += 1
        target.write_text(json.dumps(asdict(episode)) + "\n", encoding="utf-8")
        # Ordering is (mtime_ns, name) and fast appends can share a tick,
        # leaving ties to random names — force monotonic mtimes.
        ns = max(time.time_ns(), self._last_ns + 1)
        os.utime(target, ns=(ns, ns))
        self._last_ns = ns

    def append_trace(self, episode_id: str, trace: Sequence[Mapping[str, Any]]) -> None:
        """Full minute interaction for one episode, debug-only.

        One JSON object per line in `<id>.trace.jsonl`. Never injected
        into worker context; `read()`/`latest()` only use episode content.
        """
        if not trace:
            return
        target = self.dir / f"{episode_id}.trace.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for record in trace:
                handle.write(json.dumps(dict(record)) + "\n")

    def read_trace(self, episode_id: str) -> list[dict[str, Any]]:
        """Minute interaction for one episode, empty when none was stored."""
        target = self.dir / f"{episode_id}.trace.jsonl"
        if not target.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in target.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def _files(self) -> list[Path]:
        return sorted(
            (f for f in self.dir.glob("*.jsonl") if not f.name.endswith(".trace.jsonl")),
            key=lambda f: (f.stat().st_mtime_ns, f.name),
        )

    def _load(self, *, ok_only: bool) -> list[Episode]:
        if not self.dir.exists():
            return []
        episodes: list[Episode] = []
        for file in self._files():
            try:
                episode = Episode(**json.loads(file.read_text(encoding="utf-8")))
            except (ValueError, TypeError, OSError):
                continue
            if ok_only and episode.status != OK:
                continue
            episodes.append(episode)
        return episodes

    def read(self, thread: str, session: str) -> list[Episode]:
        """Retained handoffs for one thread in one session. Failed dispatches
        are excluded: a worker that died must never become context."""
        if not session:
            raise ValueError("EpisodeStore.read requires a non-empty session.")
        return [e for e in self._load(ok_only=True) if e.thread == thread and e.session == session]

    def latest(self, thread: str, session: str) -> Episode | None:
        episodes = self.read(thread, session=session)
        return episodes[-1] if episodes else None

    def names(self, session: str) -> list[str]:
        if not session:
            raise ValueError("EpisodeStore.names requires a non-empty session.")
        seen: dict[str, int] = {}
        for episode in self._load(ok_only=True):
            if episode.session != session:
                continue
            seen[episode.thread] = seen.get(episode.thread, 0) + 1
        return sorted(seen)

    def count(self, thread: str, session: str) -> int:
        return len(self.read(thread, session=session))

    def get(self, episode_id: str) -> Episode | None:
        """One episode by id (the `<id>.jsonl` filename)."""
        for episode in self._load(ok_only=False):
            if episode.id == episode_id:
                return episode
        return None

    def by_session(self, session_id: str) -> list[Episode]:
        """Every episode one orchestrator run left behind, failures included."""
        return [e for e in self._load(ok_only=False) if e.session == session_id]


def _header(index: int, episode: Episode) -> str:
    header = (
        f"\n=== Episode {index} | {episode.created_at} | action: {episode.action}"
        f" | session: {episode.session}"
    )
    return header + f" ===\n{episode.content}"


def own_history_messages(thread: str, episodes: list[Episode]) -> list[dict[str, Any]]:
    """This thread's own past as proper turns, like the orchestrator replay.

    Each prior action replays as `user`, each episode as the `assistant`
    reply — verbatim, no headers. The worker sees its own past the way it
    happened instead of one flattened `user` blob."""
    messages: list[dict[str, Any]] = []
    for episode in episodes:
        messages.append({"role": "user", "content": episode.action})
        messages.append({"role": "assistant", "content": episode.content})
    return messages


def source_message(episode: Episode) -> dict[str, Any]:
    """Latest episode of another thread, as orchestrator-provided input."""
    return {"role": "user", "content": render_source_context(episode)}


def render_source_context(episode: Episode) -> str:
    """Latest episode of another thread, injected as input for this dispatch."""
    return (
        f'Thread: "{episode.thread}" '
        f"\n\n{episode.content}"
    )


def render_thread_document(thread: str, episodes: list[Episode]) -> str:
    if not episodes:
        return f'Thread "{thread}" has no retained episodes.'
    header = f'Thread "{thread}" retained episodes ({len(episodes)} total):'
    body = [_header(i, e) for i, e in enumerate(episodes, start=1)]
    return "\n".join([header, *body])
