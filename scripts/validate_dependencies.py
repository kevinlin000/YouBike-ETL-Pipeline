from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIREMENT_FILES = (
    "requirements.txt",
    "requirements-test.txt",
    "requirements-dev.txt",
    "requirements-dbt.txt",
    "requirements_app.txt",
    "api/requirements.txt",
)

LOCKED_REQUIREMENT_FILES = {"requirements.txt"}
ALLOWED_INCLUDE_TARGETS = {"requirements-test.txt"}
VERSION_OPERATORS = ("==", ">=", "<=", "~=", "!=", ">", "<")
DISALLOWED_TOKENS = ("git+", "http://", "https://", "file:", "../", "./")


@dataclass(frozen=True)
class RequirementEntry:
    path: str
    line_number: int
    raw: str
    requirement: str


@dataclass(frozen=True)
class ValidationResult:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def strip_inline_comment(line: str) -> str:
    return line.split("#", maxsplit=1)[0].strip()


def requirement_name(requirement: str) -> str:
    return re.split(r"\s*(?:==|>=|<=|~=|!=|>|<|\[)", requirement, maxsplit=1)[0].lower()


def is_include_line(requirement: str) -> bool:
    return requirement.startswith("-r ") or requirement.startswith("--requirement ")


def include_target(requirement: str) -> str:
    return requirement.split(maxsplit=1)[1].strip()


def has_version_constraint(requirement: str) -> bool:
    return any(operator in requirement for operator in VERSION_OPERATORS)


def has_exact_pin(requirement: str) -> bool:
    return "==" in requirement


def iter_requirement_entries(path: Path) -> list[RequirementEntry]:
    entries = []
    relative_path = str(path.relative_to(PROJECT_ROOT))
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        requirement = strip_inline_comment(raw_line)
        if not requirement:
            continue
        entries.append(
            RequirementEntry(
                path=relative_path,
                line_number=line_number,
                raw=raw_line,
                requirement=requirement,
            )
        )
    return entries


def validate_entry(entry: RequirementEntry, locked_file: bool) -> list[str]:
    errors = []
    requirement = entry.requirement
    location = f"{entry.path}:{entry.line_number}"

    if is_include_line(requirement):
        target = include_target(requirement)
        if entry.path != "requirements-dev.txt":
            errors.append(f"{location} include directives are only allowed in requirements-dev.txt")
        elif target not in ALLOWED_INCLUDE_TARGETS:
            errors.append(f"{location} includes unexpected requirements file: {target}")
        return errors

    if requirement.startswith("-"):
        errors.append(f"{location} unsupported pip option in requirements file")
        return errors

    lowered = requirement.lower()
    for token in DISALLOWED_TOKENS:
        if token in lowered:
            errors.append(f"{location} direct URL, VCS, or local path dependency is not allowed")
            break

    if "*" in requirement:
        errors.append(f"{location} wildcard dependency versions are not allowed")

    if not has_version_constraint(requirement):
        errors.append(f"{location} dependency must include an explicit version constraint")
    elif locked_file and not has_exact_pin(requirement):
        errors.append(f"{location} locked requirements must use exact == pins")

    return errors


def validate(requirement_files: tuple[str, ...] = REQUIREMENT_FILES) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    for relative_path in requirement_files:
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            errors.append(f"Missing requirements file: {relative_path}")
            continue

        seen_names: dict[str, int] = {}
        locked_file = relative_path in LOCKED_REQUIREMENT_FILES
        for entry in iter_requirement_entries(path):
            errors.extend(validate_entry(entry, locked_file=locked_file))
            if is_include_line(entry.requirement) or entry.requirement.startswith("-"):
                continue
            name = requirement_name(entry.requirement)
            if name in seen_names:
                errors.append(
                    f"{entry.path}:{entry.line_number} duplicate dependency "
                    f"{name!r}; first seen on line {seen_names[name]}"
                )
            else:
                seen_names[name] = entry.line_number

    return ValidationResult(errors=errors, warnings=warnings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dependency manifest hygiene for requirements files."
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
    print("Dependency manifest validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
