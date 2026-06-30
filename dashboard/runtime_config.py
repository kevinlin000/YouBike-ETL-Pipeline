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


def runtime_value(name: str, default: str, secrets: Mapping[str, Any] | None = None) -> str:
    env_value = os.getenv(name)
    if env_value is not None:
        return env_value

    secret_value = _secret_value(secrets, name)
    if secret_value is None:
        return default
    return str(secret_value)


def runtime_bool(name: str, default: bool, secrets: Mapping[str, Any] | None = None) -> bool:
    value = runtime_value(name, str(default), secrets)
    return value.strip().lower() in TRUE_VALUES


def dashboard_api_base_url(secrets: Mapping[str, Any] | None = None) -> str:
    return runtime_value("API_BASE_URL", "http://api:8000", secrets)


def dashboard_demo_mode_default(secrets: Mapping[str, Any] | None = None) -> bool:
    return runtime_bool("DASHBOARD_DEMO_MODE", False, secrets)
