"""Tau is the engine: it owns providers, credentials, and coding tools."""

from __future__ import annotations

from pathlib import Path

from tau_coding.provider_config import load_provider_settings
from tau_coding.provider_runtime import create_model_provider
from tau_coding.tools import create_coding_tools


def build_engine(
    *,
    provider_name: str | None = None,
    model: str | None = None,
    cwd: str | Path | None = None,
):
    """Resolve a Tau provider/model and the worker tool set.

    Returns `(provider, model, worker_tools)`. Loom adds nothing here - model
    catalogs, auth, retries, and tool safety all stay in Tau.
    """
    settings = load_provider_settings()
    config = settings.get_provider(provider_name)
    resolved_model = model or config.default_model
    provider = create_model_provider(config, model=resolved_model)
    return provider, resolved_model, create_coding_tools(cwd=cwd)
