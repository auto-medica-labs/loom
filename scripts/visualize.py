"""Visualize LLM message construction for one session. (ponytail: read-only, no deps.)"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loom.episodes import EpisodeStore, own_history_messages, source_message
from loom.sessions import SessionStore


def _preview(text: str, limit: int = 160) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _specs(session_path: Path) -> dict[str, dict]:
    """Last dispatch spec per thread: {name: {action, sources}}. (ponytail: best-effort parse.)"""
    import json

    found: dict[str, dict] = {}
    for line in session_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("type") != "assistant":
            continue
        for call in record.get("tool_calls") or []:
            arg = (call.get("arguments") or {}) if isinstance(call, dict) else {}
            if isinstance(arg, str):
                try:
                    arg = json.loads(arg)
                except ValueError:
                    continue
            if not isinstance(arg, dict):
                continue
            raw = arg.get("items")
            items = raw if isinstance(raw, list) else [arg]
            for item in items:
                if isinstance(item, dict) and item.get("name"):
                    found[str(item["name"])] = {
                        "action": str(item.get("action", "")),
                        "sources": [str(s) for s in item.get("threads", []) or []],
                    }
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show orchestrator + worker messages.")
    parser.add_argument("--session", default=None, help="Session id; default: latest.")
    args = parser.parse_args(argv)

    cwd = Path(".").resolve()
    sessions, episodes = (
        SessionStore(cwd / ".loom" / "sessions"),
        EpisodeStore(cwd / ".loom" / "episodes"),
    )
    sid = args.session or (sessions.ids()[-1] if sessions.ids() else None)
    if not sid:
        print("no sessions found", file=sys.stderr)
        raise SystemExit(1)
    print(f"session: {sid} ({sessions.path_of(sid)})\n")

    msgs = sessions.messages(sid, episodes) or []
    print(f"== ORCHESTRATOR ({len(msgs)} msgs, system=orchestrator_prompt) ==")
    for i, m in enumerate(msgs):
        body = m.get("content", "")
        extra = ""
        if m.get("tool_calls"):
            calls = ", ".join(
                f"{c['function']['name']}:{str(c['function'].get('arguments', ''))[:60]}"
                for c in m["tool_calls"]
            )
            extra = f" tool_calls=[{calls}]"
        print(f"[{i}] {m['role']}{extra} ({len(str(body))} chars): {_preview(str(body))}")
    print()

    specs = _specs(sessions.path_of(sid))
    threads = episodes.names(sid)
    if not threads:
        print("(no threads yet)")
        return
    for thread in threads:
        spec = specs.get(thread, {})
        sources = spec.get("sources", [])
        action = spec.get("action", "<next action>")
        own = episodes.read(thread, session=sid)
        prior = own[:-1] if (own and action and own[-1].action == action) else own
        worker: list[dict] = [{"role": "system", "content": "worker_prompt(cwd)"}]
        worker.extend(own_history_messages(thread, prior))
        for src in sources:  # type: ignore[union-attr]
            ep = episodes.latest(src, session=sid)
            worker.append(
                source_message(ep)
                if ep
                else {"role": "user", "content": f"[missing source: {src}]"}
            )
        worker.append({"role": "user", "content": action})
        final = own[-1].content if (own and action and own[-1].action == action) else None
        if final:
            worker.append({"role": "assistant", "content": final})
        kind = "action" if action != "<next action>" else "hypothetical action"
        n_prior, n_src = len(prior) * 2, len(sources)  # type: ignore[arg-type]
        tail = " + 1 assistant (stored episode)" if final else ""
        label = f"{len(worker)} msgs: 1 system + {n_prior} prior + {n_src} sources + 1 {kind}{tail}"
        print(f"== WORKER '{thread}' ({label}) ==")
        for i, m in enumerate(worker):
            print(
                f"[{i}] {m['role']} ({len(str(m['content']))} chars): {_preview(str(m['content']))}"
            )
        print()


if __name__ == "__main__":
    main()
