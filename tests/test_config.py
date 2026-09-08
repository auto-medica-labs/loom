"""Credential file + config precedence (no .env)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from loom.engine import (
    DEFAULT_MODEL,
    build_engine,
    load_credentials,
    save_credentials,
)


def _cred_file(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "credential.json"
    monkeypatch.setenv("LOOM_CREDENTIAL_FILE", str(path))
    for var in (
        "LOOM_LLM_PROVIDER_API_KEY",
        "LOOM_LLM_PROVIDER_BASE_URL",
        "LOOM_MODEL",
        "LOOM_PROVIDER",
    ):
        monkeypatch.delenv(var, raising=False)
    return path


def test_save_load_roundtrip_and_perms(tmp_path: Path, monkeypatch) -> None:
    path = _cred_file(tmp_path, monkeypatch)
    save_credentials({"base_url": "http://x/v1", "api_key": "k", "model": "openai:gpt-4o"})
    assert load_credentials() == {
        "base_url": "http://x/v1",
        "api_key": "k",
        "model": "openai:gpt-4o",
    }
    assert path.stat().st_mode & 0o777 == 0o600


def test_missing_or_corrupt_file_returns_empty(tmp_path: Path, monkeypatch) -> None:
    path = _cred_file(tmp_path, monkeypatch)
    assert load_credentials() == {}
    path.write_text("not json{")
    assert load_credentials() == {}


def test_model_precedence_flag_over_env_over_file_over_default(tmp_path: Path, monkeypatch) -> None:
    _cred_file(tmp_path, monkeypatch)
    save_credentials({"model": "openai:from-file"})

    _, model, _ = build_engine(cwd=tmp_path)
    assert model == "openai:from-file"

    monkeypatch.setenv("LOOM_MODEL", "openai:from-env")
    _, model, _ = build_engine(cwd=tmp_path)
    assert model == "openai:from-env"

    _, model, _ = build_engine(model="openai:from-flag", cwd=tmp_path)
    assert model == "openai:from-flag"

    assert os.getenv("LOOM_MODEL") == "openai:from-env"  # file never overwrites env


def test_provider_and_model_defaults_without_file(tmp_path: Path, monkeypatch) -> None:
    _cred_file(tmp_path, monkeypatch)
    provider, model, _ = build_engine(cwd=tmp_path)
    assert provider is None
    assert model == DEFAULT_MODEL


def test_file_values_do_not_overwrite_process_env(tmp_path: Path, monkeypatch) -> None:
    _cred_file(tmp_path, monkeypatch)
    save_credentials({"api_key": "file-key"})
    monkeypatch.setenv("LOOM_LLM_PROVIDER_API_KEY", "env-key")
    build_engine(cwd=tmp_path)
    assert os.getenv("LOOM_LLM_PROVIDER_API_KEY") == "env-key"


def test_file_is_plain_json(tmp_path: Path, monkeypatch) -> None:
    path = _cred_file(tmp_path, monkeypatch)
    save_credentials({"model": "m"})
    assert json.loads(path.read_text()) == {"model": "m"}
