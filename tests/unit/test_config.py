from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from question_builder.config.models import AppConfig, load_config


@pytest.fixture
def config() -> AppConfig:
    return load_config(Path("config/default.yaml"))


def test_approved_quality_threshold_defaults(config: AppConfig) -> None:
    assert config.quality.noncritical_recognition_accept == 0.95
    assert config.quality.critical_recognition_accept == 0.98
    assert config.quality.recognition_fallback_floor == 0.90
    assert config.quality.split_accept == 0.98
    assert config.quality.answer_match_accept == 0.995
    assert config.quality.answer_match_margin == 0.10
    assert config.quality.answer_verify_accept == 0.995


def test_recognition_fallback_floor_cannot_exceed_critical_accept() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "quality": {
                    "recognition_fallback_floor": 0.99,
                    "critical_recognition_accept": 0.98,
                }
            }
        )


def test_api_keys_are_not_config_fields() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"api_key": "must-not-live-in-yaml"})


def test_cli_entrypoint_resolves() -> None:
    from question_builder.cli.app import app

    assert app.info.name == "qbuilder"


def test_cli_help_is_runnable() -> None:
    from question_builder.cli.app import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "qbuilder" in result.output.lower()
