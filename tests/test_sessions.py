"""Session transcripts: one {type, text} line per turn, in order."""

from __future__ import annotations

from pathlib import Path

from loom.sessions import SessionStore


def test_episode_lines_are_references_resolved_at_render(tmp_path: Path) -> None:
    import json

    from loom.episodes import Episode, EpisodeStore

    episodes = EpisodeStore(tmp_path / "episodes")
    episodes.append(Episode("research", "look", "found README"))
    episodes.append(Episode("signals", "look", "pyproject only"))
    first, second = episodes.read("research") + episodes.read("signals")

    sessions = SessionStore(tmp_path / "sessions")
    sid = sessions.start("what is this repo?")
    sessions.log_output(sid, "Planning: dispatch research threads.")
    sessions.log_episode_ref(sid, "research", first.id)
    sessions.log_episode_ref(sid, "signals", second.id)

    lines = sessions.path_of(sid).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["type"] for line in lines] == ["input", "output", "episode", "episode"]
    ref = json.loads(lines[2])
    assert ref == {"type": "episode", "label": "research", "id": first.id}
    assert "text" not in ref

    rendered = sessions.render(sid, episodes)
    assert rendered is not None
    assert "what is this repo?" in rendered
    assert "Planning: dispatch research threads." in rendered
    assert "== research ==\nfound README" in rendered
    assert rendered.index("found README") < rendered.index("pyproject only")
    assert sessions.ids() == [sid]


def test_render_missing_session_returns_none(tmp_path: Path) -> None:
    assert SessionStore(tmp_path / "sessions").render("nope") is None


def test_skips_corrupt_lines(tmp_path: Path) -> None:
    sessions = SessionStore(tmp_path / "sessions")
    sid = sessions.start("prompt")
    with sessions.path_of(sid).open("a", encoding="utf-8") as handle:
        handle.write("not json{\n")
    sessions.log_output(sid, "still here")

    rendered = sessions.render(sid)
    assert rendered is not None
    assert "still here" in rendered


def test_open_session_continues_single_existing(tmp_path: Path) -> None:
    from loom.cli import _open_session

    sessions = SessionStore(tmp_path / "sessions")
    sid = sessions.start("original")
    assert _open_session(sessions, [sid], "follow-up") == sid
    assert sessions.ids() == [sid]
    rendered = sessions.render(sid)
    assert rendered is not None and "follow-up" in rendered


def test_open_session_branches_otherwise(tmp_path: Path) -> None:
    from loom.cli import _open_session

    sessions = SessionStore(tmp_path / "sessions")
    sid = sessions.start("original")
    branched = _open_session(sessions, [sid, "missing"], "new run")
    assert branched != sid
    assert sessions.ids() == sorted([sid, branched])
    assert _open_session(sessions, ["missing"], "other") not in (sid, branched)


def test_resume_keeps_the_prompt(tmp_path: Path) -> None:
    from loom.cli import _parse_args

    args = _parse_args(["--session", "20260909-024251-20ffb6", "write a summarize.md"])
    assert args.session == ["20260909-024251-20ffb6"]
    assert args.resume is False
    assert args.prompt == "write a summarize.md"

    args = _parse_args(["--session", "a", "--session", "b", "do X"])
    assert args.session == ["a", "b"]
    assert args.prompt == "do X"

    args = _parse_args(["--resume", "do X"])
    assert args.resume is True
    assert args.session == []
    assert args.prompt == "do X"


def test_episode_entries_split_batch_in_order(tmp_path: Path) -> None:
    from loom.agent import ToolResult
    from loom.cli import _episode_entries

    result = ToolResult(
        text="combined",
        details={
            "episodes": [
                {"name": "a", "text": "first", "id": "aaa"},
                {"name": "b", "text": "second", "id": None},
            ]
        },
    )
    assert _episode_entries(result) == [("a", "first", "aaa"), ("b", "second", None)]
    assert _episode_entries(ToolResult(text="plain")) is None
