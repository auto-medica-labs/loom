"""Loom-native engine: provider/model resolve + worker tools. No Tau."""

from __future__ import annotations

import json
import os
from pathlib import Path

from loom.coding import create_coding_tools

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


def build_engine(
    *,
    provider_name: str | None = None,
    model: str | None = None,
    cwd: str | Path | None = None,
):
    """Return `(provider, model, worker_tools)` for any-llm.

    Precedence for provider/model/credentials: flags > `LOOM_*` env >
    `~/.loom/credential.json` (written by `loom setup`) > default.
    """
    creds = load_credentials()
    for key, env in (
        ("api_key", API_KEY_ENV),
        ("base_url", BASE_URL_ENV),
        ("model", MODEL_ENV),
        ("provider", PROVIDER_ENV),
    ):
        if value := creds.get(key):
            os.environ.setdefault(env, value)
    resolved_provider = provider_name or os.getenv(PROVIDER_ENV)
    resolved_model = model or os.getenv(MODEL_ENV) or DEFAULT_MODEL
    return resolved_provider, resolved_model, create_coding_tools(cwd=cwd)
