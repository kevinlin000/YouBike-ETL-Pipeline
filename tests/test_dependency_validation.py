from scripts import validate_dependencies


def test_validate_entry_rejects_bare_dependency():
    entry = validate_dependencies.RequirementEntry(
        path="requirements_app.txt",
        line_number=1,
        raw="fastapi",
        requirement="fastapi",
    )

    errors = validate_dependencies.validate_entry(entry, locked_file=False)

    assert errors == [
        "requirements_app.txt:1 dependency must include an explicit version constraint"
    ]


def test_validate_entry_rejects_direct_url_dependency():
    entry = validate_dependencies.RequirementEntry(
        path="requirements_app.txt",
        line_number=1,
        raw="pkg @ https://example.com/pkg.whl",
        requirement="pkg @ https://example.com/pkg.whl",
    )

    errors = validate_dependencies.validate_entry(entry, locked_file=False)

    assert errors == [
        "requirements_app.txt:1 direct URL, VCS, or local path dependency is not allowed",
        "requirements_app.txt:1 dependency must include an explicit version constraint",
    ]


def test_locked_requirement_file_requires_exact_pin():
    entry = validate_dependencies.RequirementEntry(
        path="requirements.txt",
        line_number=1,
        raw="pandas>=2.0.0",
        requirement="pandas>=2.0.0",
    )

    errors = validate_dependencies.validate_entry(entry, locked_file=True)

    assert errors == ["requirements.txt:1 locked requirements must use exact == pins"]


def test_iter_requirement_entries_strips_comments(tmp_path, monkeypatch):
    monkeypatch.setattr(validate_dependencies, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "requirements.txt"
    path.write_text(
        """
# comment
fastapi>=0.100.0  # API framework
requests>=2.0.0
""",
        encoding="utf-8",
    )

    entries = validate_dependencies.iter_requirement_entries(path)

    assert [entry.requirement for entry in entries] == [
        "fastapi>=0.100.0",
        "requests>=2.0.0",
    ]


def test_current_dependency_manifests_pass():
    result = validate_dependencies.validate()

    assert result.errors == []
