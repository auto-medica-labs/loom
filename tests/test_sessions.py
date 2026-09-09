"""Session transcripts: one {type, text} line per turn, in order."""

from __future__ import annotations

from pathlib import Path

from loom.sessions import SessionStore


def test_episode_lines_are_references_resolved_in_messages(tmp_path: Path) -> None:
    import json

    from loom.episodes import Episode, EpisodeStore

    episodes = EpisodeStore(tmp_path / "episodes")
    episodes.append(Episode("research", "look", "found README"))
    episodes.append(Episode("signals", "look", "pyproject only"))
    first, second = episodes.read("research") + episodes.read("signals")

    sessions = SessionStore(tmp_path / "sessions")
    sid = sessions.start("what is this repo?")
    sessions.log_output(sid, "Planning: dispatch research threads.")
    sessions.log_episode_ref(sid, "research", first.id, "call_1")
    sessions.log_episode_ref(sid, "signals", second.id, "call_2")

    lines = sessions.path_of(sid).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["type"] for line in lines] == ["input", "output", "episode", "episode"]
    ref = json.loads(lines[2])
    assert ref == {"type": "episode", "label": "research", "id": first.id, "tool_call_id": "call_1"}
    assert "text" not in ref

    replayed = sessions.messages(sid, episodes)
    assert replayed is not None
    assert {"role": "user", "content": "what is this repo?"} in replayed
    assert {"role": "assistant", "content": "Planning: dispatch research threads."} in replayed
    tools = [m for m in replayed if m.get("role") == "tool"]
    assert [t["content"] for t in tools] == ["found README", "pyproject only"]
    assert sessions.ids() == [sid]


def test_skips_corrupt_lines(tmp_path: Path) -> None:
    sessions = SessionStore(tmp_path / "sessions")
    sid = sessions.start("prompt")
    with sessions.path_of(sid).open("a", encoding="utf-8") as handle:
        handle.write("not json{\n")
    sessions.log_output(sid, "still here")

    replayed = sessions.messages(sid)
    assert replayed is not None
    assert {"role": "assistant", "content": "still here"} in replayed


def test_open_session_continues_single_existing(tmp_path: Path) -> None:
    from loom.cli import _open_session

    sessions = SessionStore(tmp_path / "sessions")
    sid = sessions.start("original")
    assert _open_session(sessions, [sid], "follow-up") == sid
    assert sessions.ids() == [sid]
    replayed = sessions.messages(sid)
    assert replayed is not None
    assert {"role": "user", "content": "follow-up"} in replayed


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


def test_messages_replays_native_roles(tmp_path: Path) -> None:
    import json

    from loom.episodes import Episode, EpisodeStore
    from loom.sessions import SessionStore

    episodes = EpisodeStore(tmp_path / "episodes")
    episodes.append(Episode("impl/auth", "add JWT", "added middleware + tests"))
    (episode,) = episodes.read("impl/auth")

    sessions = SessionStore(tmp_path / "sessions")
    sid = sessions.start("add auth")
    sessions.log_output(sid, "Planning: research first.")
    sessions.log_assistant(
        sid, "", [{"id": "call_1", "name": "thread", "arguments": {"name": "impl/auth"}}]
    )
    sessions.log_episode_ref(sid, "impl/auth", episode.id, tool_call_id="call_1")

    assert sessions.messages(sid, episodes) == [
        {"role": "user", "content": "add auth"},
        {"role": "assistant", "content": "Planning: research first."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "thread",
                        "arguments": json.dumps({"name": "impl/auth"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "impl/auth",
            "content": "added middleware + tests",
        },
    ]


def test_messages_groups_batch_episodes_into_one_tool_message(tmp_path: Path) -> None:
    from loom.episodes import Episode, EpisodeStore
    from loom.sessions import SessionStore

    episodes = EpisodeStore(tmp_path / "episodes")
    episodes.append(Episode("a", "look", "first"))
    episodes.append(Episode("b", "look", "second"))
    (first,) = episodes.read("a")
    (second,) = episodes.read("b")

    sessions = SessionStore(tmp_path / "sessions")
    sid = sessions.start("explore")
    sessions.log_assistant(sid, "", [{"id": "call_b", "name": "thread_batch", "arguments": {}}])
    sessions.log_episode_ref(sid, "a", first.id, tool_call_id="call_b")
    sessions.log_episode_ref(sid, "b", second.id, tool_call_id="call_b")

    replayed = sessions.messages(sid, episodes)
    assert replayed is not None
    assert replayed[-1] == {
        "role": "tool",
        "tool_call_id": "call_b",
        "name": "a",
        "content": "first\n\nsecond",
    }


def test_messages_splits_episodes_with_different_call_ids(tmp_path: Path) -> None:
    from loom.episodes import Episode, EpisodeStore
    from loom.sessions import SessionStore

    episodes = EpisodeStore(tmp_path / "episodes")
    episodes.append(Episode("a", "look", "first"))
    episodes.append(Episode("b", "look", "second"))
    episodes.append(Episode("c", "look", "third"))
    (first,) = episodes.read("a")
    (second,) = episodes.read("b")
    (third,) = episodes.read("c")

    sessions = SessionStore(tmp_path / "sessions")
    sid = sessions.start("explore")
    sessions.log_assistant(
        sid, "working", [{"id": "call_b", "name": "thread_batch", "arguments": {}}]
    )
    sessions.log_episode_ref(sid, "a", first.id, tool_call_id="call_b")
    sessions.log_episode_ref(sid, "b", second.id, tool_call_id="call_b")
    sessions.log_episode_ref(sid, "c", third.id, tool_call_id="call_c")
    sessions.log_output(sid, "done")

    replayed = sessions.messages(sid, episodes)
    assert replayed is not None
    tools = [m for m in replayed if m.get("role") == "tool"]
    assert tools == [
        {
            "role": "tool",
            "tool_call_id": "call_b",
            "name": "a",
            "content": "first\n\nsecond",
        },
        {"role": "tool", "tool_call_id": "call_c", "name": "c", "content": "third"},
    ]


def test_messages_trims_dangling_tool_turn(tmp_path: Path) -> None:
    from loom.sessions import SessionStore

    sessions = SessionStore(tmp_path / "sessions")
    sid = sessions.start("do it")
    sessions.log_output(sid, "ok")
    sessions.log_assistant(sid, "", [{"id": "call_9", "name": "thread", "arguments": {}}])

    assert sessions.messages(sid) == [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "ok"},
    ]


def test_messages_skips_episode_ref_without_call_id(tmp_path: Path) -> None:
    import json

    from loom.episodes import Episode, EpisodeStore
    from loom.sessions import SessionStore

    episodes = EpisodeStore(tmp_path / "episodes")
    episodes.append(Episode("research", "look", "found README"))
    (episode,) = episodes.read("research")

    sessions = SessionStore(tmp_path / "sessions")
    sid = sessions.start("what is this repo?")
    sessions.log_output(sid, "Planning.")
    # Hand-written ref with no tool_call_id: cannot pair with an assistant
    # turn, so it is skipped, not replayed.
    with sessions.path_of(sid).open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"type": "episode", "label": "research", "id": episode.id}) + "\n"
        )

    assert sessions.messages(sid, episodes) == [
        {"role": "user", "content": "what is this repo?"},
        {"role": "assistant", "content": "Planning."},
    ]


def test_messages_missing_session_returns_none(tmp_path: Path) -> None:
    from loom.sessions import SessionStore

    assert SessionStore(tmp_path / "sessions").messages("nope") is None
