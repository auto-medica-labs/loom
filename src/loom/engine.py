"""Provider/model resolve + worker tools."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from loom.coding import CodingToolkit

DEFAULT_MODEL = "openai:gpt-5.4"
API_KEY_ENV = "LOOM_LLM_PROVIDER_API_KEY"
BASE_URL_ENV = "LOOM_LLM_PROVIDER_BASE_URL"
MODEL_ENV = "LOOM_MODEL"
PROVIDER_ENV = "LOOM_PROVIDER"


def credential_path() -> Path:
    """Home credential file. Overridable via LOOM_CREDENTIAL_FILE (tests)."""
    override = os.getenv("LOOM_CREDENTIAL_FILE")
    if override:
        return Path(override)
    return Path.home() / ".loom" / "credential.json"


def load_credentials() -> dict[str, str]:
    """Read the home credential file. Missing/corrupt → {} (setup overwrites)."""
    try:
        data = json.loads(credential_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: str(v) for k, v in data.items() if isinstance(v, str) and v}


def save_credentials(data: dict[str, str]) -> Path:
    """Write the home credential file with owner-only permissions."""
    path = credential_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return path


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Resolved provider/model/credentials plus the worker toolkit."""

    provider: str | None
    model: str
    api_key: str | None
    api_base: str | None
    coding: CodingToolkit


def build_engine(
    *,
    provider_name: str | None = None,
    model: str | None = None,
    cwd: str | Path | None = None,
) -> EngineConfig:
    """Resolve provider/model/credentials for any-llm.

    Precedence: flags > `LOOM_*` env > `~/.loom/credential.json` > default.
    Returns explicit values rather than mutating `os.environ`, so a credential
    file's API key never leaks into worker subprocesses via the environment.
    """
    creds = load_credentials()
    api_key = os.getenv(API_KEY_ENV) or creds.get("api_key")
    api_base = os.getenv(BASE_URL_ENV) or creds.get("base_url")
    resolved_provider = provider_name or os.getenv(PROVIDER_ENV) or creds.get("provider")
    resolved_model = model or os.getenv(MODEL_ENV) or creds.get("model") or DEFAULT_MODEL
    return EngineConfig(
        provider=resolved_provider,
        model=resolved_model,
        api_key=api_key,
        api_base=api_base,
        coding=CodingToolkit(cwd),
    )
