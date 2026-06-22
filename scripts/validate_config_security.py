from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"
CONFIG_SECURITY_DOC_PATH = PROJECT_ROOT / "docs" / "configuration_security.md"

FORBIDDEN_TRACKED_PATHS = {
    ".env",
    "analytics/dbt/profiles.yml",
    "analytics/dbt/.user.yml",
}

REQUIRED_GITIGNORE_ENTRIES = {
    ".env",
    ".venv-dbt/",
    "analytics/dbt/target/",
    "analytics/dbt/dbt_packages/",
    "analytics/dbt/logs/",
    "analytics/dbt/profiles.yml",
    "analytics/dbt/.user.yml",
}

ALLOWED_SECRET_CONTEXTS = (
    "your_",
    "dummy",
    "example",
    "placeholder",
    "secret_key_change_in_production",
    "root",
    "youbike",
    "admin",
    "${",
    "{{",
    "os.getenv(",
    "env_var(",
    "_db_password_env",
    "db_password",
    "get_gcp_secret(",
    "api-demo-fixtures",
    "sha256:",
    "portfolio-demo",
    "contract-check",
)

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\bgh[pousr]_[0-9A-Za-z_]{36,}\b"),
    re.compile(r'"private_key"\s*:\s*"[^"]+BEGIN PRIVATE KEY'),
    re.compile(r'"type"\s*:\s*"service_account"'),
    re.compile(
        r"(?i)(password|secret|token|api[_-]?key)\s*[:=]\s*['\"]?([^'\"\s#][^'\"\n#]*)"
    ),
)

TEXT_FILE_SUFFIXES = {
    ".env",
    ".example",
    ".http",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sql",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class ValidationResult:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def run_git_ls_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def is_text_candidate(path: str) -> bool:
    file_path = Path(path)
    if file_path.suffix in TEXT_FILE_SUFFIXES:
        return True
    return file_path.name in {".env.example", ".gitignore", "Makefile"}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_env_example_names(content: str) -> set[str]:
    names = set()
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            stripped = stripped[1:].strip()
        if "=" not in stripped:
            continue
        name = stripped.split("=", maxsplit=1)[0].strip()
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            names.add(name)
    return names


def secret_match_is_allowed(line: str) -> bool:
    lowered = line.lower()
    return any(token.lower() in lowered for token in ALLOWED_SECRET_CONTEXTS)


def find_secret_like_lines(path: Path, content: str) -> list[str]:
    findings = []
    try:
        display_path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = path.name
    for line_number, line in enumerate(content.splitlines(), start=1):
        if secret_match_is_allowed(line):
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(f"{display_path}:{line_number}")
                break
    return findings


def validate(tracked_files: list[str] | None = None) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    tracked = tracked_files if tracked_files is not None else run_git_ls_files()
    tracked_set = set(tracked)

    forbidden_tracked = sorted(FORBIDDEN_TRACKED_PATHS & tracked_set)
    for path in forbidden_tracked:
        errors.append(f"Sensitive local file is tracked: {path}")

    if not GITIGNORE_PATH.exists():
        errors.append(".gitignore is missing")
    else:
        gitignore = read_text(GITIGNORE_PATH)
        for entry in sorted(REQUIRED_GITIGNORE_ENTRIES):
            if entry not in gitignore:
                errors.append(f".gitignore missing required entry: {entry}")

    if not ENV_EXAMPLE_PATH.exists():
        errors.append(".env.example is missing")
        env_names: set[str] = set()
    else:
        env_example = read_text(ENV_EXAMPLE_PATH)
        env_names = parse_env_example_names(env_example)
        if "AIRFLOW_PASSWORD=admin" in env_example and "正式部署不要使用預設帳密" not in env_example:
            errors.append(".env.example keeps Airflow default password without a production warning")

    if not CONFIG_SECURITY_DOC_PATH.exists():
        errors.append("docs/configuration_security.md is missing")
    else:
        config_doc = read_text(CONFIG_SECURITY_DOC_PATH)
        for name in sorted(env_names):
            if name not in config_doc:
                errors.append(f"docs/configuration_security.md missing env var: {name}")

    for path in tracked:
        if not is_text_candidate(path):
            continue
        file_path = PROJECT_ROOT / path
        if not file_path.exists():
            continue
        try:
            content = read_text(file_path)
        except UnicodeDecodeError:
            warnings.append(f"Skipped non-UTF-8 tracked file: {path}")
            continue
        for finding in find_secret_like_lines(file_path, content):
            errors.append(f"Potential secret in tracked file: {finding}")

    return ValidationResult(errors=errors, warnings=warnings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate repository config and secret-handling boundaries."
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    result = validate()
    for warning in result.warnings:
        print(f"warning: {warning}")
    if result.errors:
        for error in result.errors:
            print(f"error: {error}")
        return 1
    print("Config/security validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
