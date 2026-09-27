"""Coding toolkit: cwd jail, edit uniqueness, truncation, bash env scrub."""

from __future__ import annotations

import asyncio
from pathlib import Path

from loom.coding import CodingToolkit


def run(tool, args: dict):
    return asyncio.run(tool(args))


# --- read ---


def test_read_returns_content(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    result = run(CodingToolkit(tmp_path).read, {"path": "a.txt"})
    assert not result.is_error
    assert result.text == "hello\nworld"  # splitlines() drops the trailing newline


def test_read_rejects_relative_escape(tmp_path: Path) -> None:
    result = run(CodingToolkit(tmp_path).read, {"path": "../secret.txt"})
    assert result.is_error
    assert "escapes" in result.text


def test_read_rejects_absolute_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("s", encoding="utf-8")
    result = run(CodingToolkit(tmp_path).read, {"path": str(outside)})
    assert result.is_error
    assert "escapes" in result.text


def test_read_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("s", encoding="utf-8")
    (tmp_path / "link").symlink_to(outside)
    result = run(CodingToolkit(tmp_path).read, {"path": "link"})
    assert result.is_error
    assert "escapes" in result.text


def test_read_reports_missing_and_directory(tmp_path: Path) -> None:
    kit = CodingToolkit(tmp_path)
    assert run(kit.read, {"path": "nope.txt"}).is_error
    (tmp_path / "dir").mkdir()
    assert run(kit.read, {"path": "dir"}).is_error


# --- write ---


def test_write_creates_parents_and_writes(tmp_path: Path) -> None:
    result = run(CodingToolkit(tmp_path).write, {"path": "a/b/c.txt", "content": "hi"})
    assert not result.is_error
    assert (tmp_path / "a/b/c.txt").read_text(encoding="utf-8") == "hi"


def test_write_rejects_escape(tmp_path: Path) -> None:
    result = run(CodingToolkit(tmp_path).write, {"path": "../x.txt", "content": "y"})
    assert result.is_error


# --- edit ---


def test_edit_applies_exact_once(tmp_path: Path) -> None:
    path = tmp_path / "f.py"
    path.write_text("a b a\n", encoding="utf-8")
    result = run(
        CodingToolkit(tmp_path).edit,
        {"path": "f.py", "edits": [{"oldText": "a b a", "newText": "z"}]},
    )
    assert not result.is_error
    assert path.read_text(encoding="utf-8") == "z\n"


def test_edit_requires_unique_match(tmp_path: Path) -> None:
    path = tmp_path / "f.py"
    path.write_text("dup dup\n", encoding="utf-8")
    result = run(
        CodingToolkit(tmp_path).edit,
        {"path": "f.py", "edits": [{"oldText": "dup", "newText": "x"}]},
    )
    assert result.is_error
    assert "matches 2x" in result.text


# --- bash ---


def test_bash_scrubs_secret_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOOM_LLM_PROVIDER_API_KEY", "sekret")
    result = run(CodingToolkit(tmp_path).bash, {"command": "echo key=$LOOM_LLM_PROVIDER_API_KEY"})
    assert not result.is_error
    assert "sekret" not in result.text


def test_bash_truncates_large_output(tmp_path: Path) -> None:
    result = run(CodingToolkit(tmp_path).bash, {"command": "python -c \"print('x'*200000)\""})
    assert "truncated" in result.text
    assert len(result.text) < 200000


def test_bash_nonzero_exit_is_error(tmp_path: Path) -> None:
    result = run(CodingToolkit(tmp_path).bash, {"command": "exit 3"})
    assert result.is_error
    assert "[exit 3]" in result.text
