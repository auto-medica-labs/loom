"""Durable episode storage: the only thing a thread leaves behind."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
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
    session: str = ""


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

    def _files(self) -> list[Path]:
        return sorted(self.dir.glob("*.jsonl"), key=lambda f: (f.stat().st_mtime_ns, f.name))

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

    def read(self, thread: str) -> list[Episode]:
        """Retained handoffs for one thread. Failed dispatches are excluded: a
        worker that died must never become context for the next one."""
        return [e for e in self._load(ok_only=True) if e.thread == thread]

    def latest(self, thread: str) -> Episode | None:
        episodes = self.read(thread)
        return episodes[-1] if episodes else None

    def names(self) -> list[str]:
        seen: dict[str, int] = {}
        for episode in self._load(ok_only=True):
            seen[episode.thread] = seen.get(episode.thread, 0) + 1
        return sorted(seen)

    def count(self, thread: str) -> int:
        return len(self.read(thread))

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
    header = f"\n=== Episode {index} | {episode.created_at} | action: {episode.action}"
    if episode.session:
        header += f" | session: {episode.session}"
    return header + f" ===\n{episode.content}"


def render_self_context(thread: str, episodes: list[Episode]) -> str:
    """This thread's own history, injected as the worker's memory."""
    rendered = [f'Retained history for thread "{thread}":']
    rendered.extend(_header(index, episode) for index, episode in enumerate(episodes, start=1))
    return "\n".join(rendered)


def render_source_context(episode: Episode) -> str:
    """Latest episode of another thread, injected as input for this dispatch."""
    return (
        f'Latest retained episode from thread "{episode.thread}" '
        f"| {episode.created_at} | action: {episode.action}\n{episode.content}"
    )


def render_thread_document(thread: str, episodes: list[Episode]) -> str:
    if not episodes:
        return f'Thread "{thread}" has no retained episodes.'
    header = f'Thread "{thread}" retained episodes ({len(episodes)} total):'
    body = [_header(i, e) for i, e in enumerate(episodes, start=1)]
    return "\n".join([header, *body])
