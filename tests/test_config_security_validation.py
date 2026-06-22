from scripts import validate_config_security


def test_parse_env_example_names_includes_commented_variables():
    content = """
MYSQL_PASSWORD=your_app_password_here
# API_DEMO_MODE=false
# Not a variable
lowercase=value
"""

    names = validate_config_security.parse_env_example_names(content)

    assert names == {"MYSQL_PASSWORD", "API_DEMO_MODE"}


def test_find_secret_like_lines_flags_realistic_tokens(tmp_path):
    path = tmp_path / "settings.py"
    key_name = "API_" + "TO" + "KEN"
    value = "g" + "hp_" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL"
    content = "\n".join(
        [
            'PASSWORD = "your_placeholder"',
            f'{key_name} = "{value}"',
        ]
    )

    findings = validate_config_security.find_secret_like_lines(path, content)

    assert findings == ["settings.py:2"]


def test_current_repo_config_security_validation_passes():
    result = validate_config_security.validate()

    assert result.errors == []
