from __future__ import annotations

import importlib
from pathlib import Path

from typer.testing import CliRunner

from question_builder.pipeline.orchestrator import RunStatus, RunSummary


runner = CliRunner()
app_module = importlib.import_module("question_builder.cli.app")


def _summary(tmp_path: Path, *, run_id: str = "run_123") -> RunSummary:
    return RunSummary(
        run_id=run_id,
        status=RunStatus.COMPLETED,
        completed_stages=(
            "ingest",
            "parse",
            "recognition",
            "understanding",
            "split",
            "answer_extract",
            "match_verify",
            "metadata",
            "quality",
            "export_report",
        ),
        next_stage=None,
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
    )


def test_config_validate_accepts_default_config() -> None:
    result = runner.invoke(
        app_module.app,
        ["config", "validate", "--config", "config/default.yaml"],
    )

    assert result.exit_code == 0
    assert "Configuration valid" in result.output


def test_config_validate_hides_secret_like_invalid_values(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("api_key: TOP-SECRET\n", encoding="utf-8")

    result = runner.invoke(
        app_module.app,
        ["config", "validate", "--config", str(config_path)],
    )

    assert result.exit_code != 0
    assert "Configuration invalid" in result.output
    assert "TOP-SECRET" not in result.output


def test_run_rejects_missing_input_before_pipeline_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    called = False

    def build_pipeline():
        nonlocal called
        called = True
        raise AssertionError("pipeline must not be built for invalid input")

    monkeypatch.setattr(app_module, "_build_pipeline", build_pipeline)
    result = runner.invoke(
        app_module.app,
        [
            "run",
            "--input",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "output"),
            "--config",
            "config/default.yaml",
        ],
    )

    assert result.exit_code != 0
    assert called is False


def test_run_resume_and_report_all_delegate_to_question_builder_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "paper.docx").write_bytes(b"docx")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    calls: list[tuple[str, object]] = []
    summary = _summary(tmp_path)

    class FakePipeline:
        async def run(self, *, input_dir: Path, output_dir: Path, config_path: Path) -> RunSummary:
            calls.append(("run", (input_dir, output_dir, config_path)))
            return summary

        async def resume(self, run_id: str) -> RunSummary:
            calls.append(("resume", run_id))
            return summary

        async def report(self, run_id: str) -> RunSummary:
            calls.append(("report", run_id))
            return summary

    fake = FakePipeline()
    monkeypatch.setattr(app_module, "_build_pipeline", lambda: fake)

    run_result = runner.invoke(
        app_module.app,
        [
            "run",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--config",
            str(config_path),
        ],
    )
    resume_result = runner.invoke(app_module.app, ["resume", "run_123"])
    report_result = runner.invoke(app_module.app, ["report", "run_123"])

    assert run_result.exit_code == 0
    assert resume_result.exit_code == 0
    assert report_result.exit_code == 0
    assert calls == [
        ("run", (input_dir, output_dir, config_path)),
        ("resume", "run_123"),
        ("report", "run_123"),
    ]
    assert "run_123" in run_result.output
    assert "run_123" in resume_result.output
    assert "run_123" in report_result.output
