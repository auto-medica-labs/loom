"""Durable episode storage: the only thing a thread leaves behind."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

OK = "ok"
ERROR = "error"
TIMED_OUT = "timed_out"
CANCELLED = "cancelled"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Episode:
    """One completed dispatch of one thread.

    `content` is the worker's final response, verbatim. `status` is `ok` for a
    handoff; anything else records a dispatch that died before answering.
    """

    thread: str
    action: str
    content: str
    status: str = OK
    created_at: str = field(default_factory=_now)


class EpisodeStore:
    """Append-only JSONL store for one session's threads.

    Flat file on purpose: episodes are write-once, read-many, and never
    queried except by thread name.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, episode: Episode) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(episode)) + "\n")

    def _load(self, *, ok_only: bool) -> list[Episode]:
        if not self.path.exists():
            return []
        episodes: list[Episode] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            episode = Episode(**json.loads(line))
            if ok_only and episode.status != OK:
                continue
            episodes.append(episode)
        return episodes

    def read(self, thread: str) -> list[Episode]:
        """Retained handoffs for one thread. Failed dispatches are excluded: a
        worker that died must never become context for the next one."""
        return [e for e in self._load(ok_only=True) if e.thread == thread]

    def own(self, thread: str) -> list[Episode]:
        return self.read(thread)

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


def render_self_context(thread: str, episodes: list[Episode]) -> str:
    """This thread's own history, injected as the worker's memory."""
    rendered = [f'Retained history for thread "{thread}":']
    for index, episode in enumerate(episodes, start=1):
        rendered.append(
            f"\n=== Episode {index} | {episode.created_at} | action: {episode.action} ===\n"
            f"{episode.content}"
        )
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
    body = [
        f"\n=== Episode {i} | {e.created_at} | action: {e.action} ===\n{e.content}"
        for i, e in enumerate(episodes, start=1)
    ]
    return "\n".join([header, *body])
