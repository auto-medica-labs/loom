"""Loom-native coding tools. No Tau dependency.

Same 4 tool names/schemas as Tau (`read`/`write`/`edit`/`bash`) so worker
prompts keep working. Text-only, cwd-jailed, truncated.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loom.agent import Tool, ToolResult

MAX_BYTES = 50 * 1024
MAX_LINES = 2_000

_locks: dict[Path, asyncio.Lock] = {}


def _lock(path: Path) -> asyncio.Lock:
    lock = _locks.get(path)
    if lock is None:
        lock = asyncio.Lock()
        _locks[path] = lock
    return lock


def _resolve(cwd: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("path must be a non-empty string")
    p = Path(raw)
    resolved = (cwd / p).resolve() if not p.is_absolute() else p.resolve()
    # ponytail: cwd jail is security, never simplify away
    if resolved != cwd.resolve() and cwd.resolve() not in resolved.parents:
        raise ValueError(f"path escapes working directory: {raw}")
    return resolved


def _truncate(text: str) -> str:
    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        lines = lines[-MAX_LINES:]
        return (
            "\n".join(lines)
            + f"\n\n[{len(text.splitlines())} lines total, showing last {MAX_LINES}]"
        )
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) > MAX_BYTES:
        cut = text.encode("utf-8", errors="replace")[-MAX_BYTES:].decode("utf-8", errors="replace")
        return cut + f"\n\n[output truncated to last {MAX_BYTES // 1024}KB]"
    return text


def create_coding_tools(cwd: str | Path | None = None) -> list[Tool]:
    root = Path(cwd).resolve() if cwd else Path.cwd().resolve()

    async def read_fn(args: dict[str, Any]) -> ToolResult:
        try:
            path = _resolve(root, args.get("path"))
        except ValueError as exc:
            return ToolResult(text=f"Error: {exc}", is_error=True)
        if not path.exists():
            return ToolResult(text=f"Error: file not found: {path}", is_error=True)
        if path.is_dir():
            return ToolResult(text=f"Error: path is a directory: {path}", is_error=True)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(text=f"Error: not a UTF-8 text file: {path}", is_error=True)
        lines = text.splitlines()
        offset = args.get("offset")
        limit = args.get("limit")
        start = max(int(offset) - 1, 0) if isinstance(offset, int) and offset > 0 else 0
        end = start + int(limit) if isinstance(limit, int) and limit > 0 else None
        return ToolResult(text=_truncate("\n".join(lines[start:end])) or "(empty file)")

    async def write_fn(args: dict[str, Any]) -> ToolResult:
        content = args.get("content")
        if not isinstance(content, str):
            return ToolResult(text="Error: content must be a string", is_error=True)
        try:
            path = _resolve(root, args.get("path"))
        except ValueError as exc:
            return ToolResult(text=f"Error: {exc}", is_error=True)
        async with _lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return ToolResult(text=f"Successfully wrote to {path}.")

    async def edit_fn(args: dict[str, Any]) -> ToolResult:
        edits = args.get("edits")
        if not isinstance(edits, list) or not edits:
            return ToolResult(text="Error: edits must be a non-empty list", is_error=True)
        try:
            path = _resolve(root, args.get("path"))
        except ValueError as exc:
            return ToolResult(text=f"Error: {exc}", is_error=True)
        if not path.exists() or path.is_dir():
            return ToolResult(text=f"Error: file not found: {path}", is_error=True)
        async with _lock(path):
            content = path.read_text(encoding="utf-8")
            for i, edit in enumerate(edits):
                e: Any = edit
                if not isinstance(e, dict):
                    return ToolResult(text=f"Error: edit {i} must be an object", is_error=True)
                old, new = e.get("oldText"), e.get("newText")
                if not isinstance(old, str) or not old or not isinstance(new, str):
                    return ToolResult(
                        text=f"Error: edit {i} needs non-empty oldText + newText", is_error=True
                    )
                n = content.count(old)
                if n != 1:
                    return ToolResult(
                        text=f"Error: edit {i} matches {n}x (need 1)",
                        is_error=True,
                    )
                content = content.replace(old, new, 1)
            path.write_text(content, encoding="utf-8")
        return ToolResult(text=f"Successfully edited {path} ({len(edits)} edits).")

    async def bash_fn(args: dict[str, Any]) -> ToolResult:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolResult(text="Error: command must be a non-empty string", is_error=True)
        timeout = args.get("timeout")
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            return ToolResult(text="Error: timeout must be > 0", is_error=True)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=root,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                out, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=float(timeout) if timeout else 120.0
                )
            except TimeoutError:
                proc.kill()
                return ToolResult(text="Error: command timed out", is_error=True)
        except Exception as exc:
            return ToolResult(text=f"Error: {exc}", is_error=True)
        text_out = _truncate(out.decode(errors="replace")) or "(no output)"
        if proc.returncode != 0:
            return ToolResult(text=f"{text_out}\n\n[exit {proc.returncode}]", is_error=True)
        return ToolResult(text=text_out)

    return [
        Tool(
            name="read",
            description="Read a UTF-8 text file. Use offset/limit for large files.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
            execute_fn=read_fn,
        ),
        Tool(
            name="write",
            description="Write content to a file. Creates parents, overwrites existing.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            execute_fn=write_fn,
        ),
        Tool(
            name="edit",
            description="Exact oldText->newText replacement. Each oldText must match exactly once.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {"type": "string"},
                                "newText": {"type": "string"},
                            },
                            "required": ["oldText", "newText"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
            execute_fn=edit_fn,
        ),
        Tool(
            name="bash",
            description="Run a shell command in the working directory.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "number"},
                },
                "required": ["command"],
            },
            execute_fn=bash_fn,
        ),
    ]


__all__ = ["create_coding_tools"]
