from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "on"}


def _secret_value(secrets: Mapping[str, Any] | None, name: str) -> Any:
    if secrets is None:
        return None
    try:
        return secrets.get(name)
    except Exception:
        return None


def _configured_value(name: str, secrets: Mapping[str, Any] | None = None) -> str | None:
    env_value = os.getenv(name)
    if env_value is not None:
        return env_value

    secret_value = _secret_value(secrets, name)
    if secret_value is None:
        return None
    return str(secret_value)


def runtime_value(name: str, default: str, secrets: Mapping[str, Any] | None = None) -> str:
    configured_value = _configured_value(name, secrets)
    if configured_value is None:
        return default
    return configured_value


def runtime_bool(name: str, default: bool, secrets: Mapping[str, Any] | None = None) -> bool:
    value = runtime_value(name, str(default), secrets)
    return value.strip().lower() in TRUE_VALUES


def dashboard_api_base_url(secrets: Mapping[str, Any] | None = None) -> str:
    return runtime_value("API_BASE_URL", "http://api:8000", secrets)


def dashboard_demo_mode_default(secrets: Mapping[str, Any] | None = None) -> bool:
    configured_demo_mode = _configured_value("DASHBOARD_DEMO_MODE", secrets)
    if configured_demo_mode is not None:
        return configured_demo_mode.strip().lower() in TRUE_VALUES

    # Public Streamlit deployments usually do not have a backend API.
    # Docker Compose sets API_BASE_URL explicitly, so it still defaults to live mode there.
    return _configured_value("API_BASE_URL", secrets) is None
