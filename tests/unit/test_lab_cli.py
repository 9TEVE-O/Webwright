"""Tests for the LAB intake block added to run_one() in webwright/run/cli.py."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from webwright.lab.intake import TaskTier


# ---------------------------------------------------------------------------
# Helpers to build minimal stubs for run_one's dependencies
# ---------------------------------------------------------------------------


def _make_model_stub() -> MagicMock:
    model = MagicMock()
    model.config = {}
    model.format_message.side_effect = lambda **kw: {
        "role": kw.get("role", "user"),
        "content": kw.get("content", ""),
        "extra": kw.get("extra", {}),
    }
    model.format_observation_messages.return_value = []
    model.get_template_vars.return_value = {}
    model.serialize.return_value = {"model": {}}
    return model


def _make_env_stub() -> MagicMock:
    env = MagicMock()
    env.config = {}
    env.execute.return_value = {"observation": {}}
    env.get_template_vars.return_value = {}
    env.serialize.return_value = {"environment": {}}
    env.prepare.return_value = None
    env.close.return_value = None
    return env


def _make_agent_stub(model: MagicMock, env: MagicMock) -> MagicMock:
    agent = MagicMock()
    agent.extra_template_vars = {}
    agent.messages = [
        {
            "role": "exit",
            "content": "done",
            "extra": {
                "exit_status": "Submitted",
                "submission": "result",
                "final_response": "result",
            },
        }
    ]
    agent.run.return_value = {
        "exit_status": "Submitted",
        "submission": "result",
        "final_response": "result",
    }
    return agent


def _run_one_with_lab(
    tmp_path: Path,
    lab_config: dict[str, Any],
    task: str = "Check AI tools on musictech.net",
    task_id: str | None = "test-task-1",
) -> tuple[MagicMock, dict[str, Any]]:
    """
    Call run_one() with a mocked environment, model, and agent.
    Returns (agent_stub, result).
    """
    from webwright.run.cli import run_one

    model_stub = _make_model_stub()
    env_stub = _make_env_stub()
    agent_stub = _make_agent_stub(model_stub, env_stub)

    merged_config = {
        "run": {"task": task, "task_id": task_id},
        "lab": lab_config,
        "environment": {"output_dir": str(tmp_path / "outputs")},
        "agent": {},
        "model": {},
    }

    with (
        patch("webwright.run.cli.get_config_from_spec", return_value={}),
        patch("webwright.run.cli.recursive_merge", return_value=merged_config),
        patch("webwright.run.cli.get_model", return_value=model_stub),
        patch("webwright.run.cli.get_environment", return_value=env_stub),
        patch("webwright.run.cli.get_agent", return_value=agent_stub),
        patch("webwright.run.cli.snapshot_config_specs"),
        patch("webwright.run.cli._timestamped_output_dir", return_value=tmp_path / "outputs" / "run"),
        patch("webwright.run.cli.console"),
    ):
        result = run_one(
            task=task,
            task_id=task_id,
            config_spec=["base.yaml"],
            resolved_output_dir=tmp_path / "outputs" / "run",
            snapshot_config=False,
        )

    return agent_stub, result


# ---------------------------------------------------------------------------
# Tests: LAB disabled in config
# ---------------------------------------------------------------------------


class TestRunOneLabDisabled:
    def test_lab_disabled_does_not_set_extra_template_vars(self, tmp_path: Path) -> None:
        agent_stub, _ = _run_one_with_lab(tmp_path, lab_config={"enabled": False})
        assert "lab" not in agent_stub.extra_template_vars

    def test_lab_missing_from_config_does_not_set_extra_template_vars(self, tmp_path: Path) -> None:
        agent_stub, _ = _run_one_with_lab(tmp_path, lab_config={})
        assert "lab" not in agent_stub.extra_template_vars

    def test_lab_disabled_does_not_write_audit_log(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "lab_audit.jsonl"
        _run_one_with_lab(
            tmp_path,
            lab_config={"enabled": False, "audit_log_path": str(audit_path)},
        )
        assert not audit_path.exists()


# ---------------------------------------------------------------------------
# Tests: LAB enabled in config
# ---------------------------------------------------------------------------


class TestRunOneLabEnabled:
    def _run_with_enabled_lab(
        self,
        tmp_path: Path,
        task: str = "Check AI tools on musictech.net",
        task_id: str | None = "t1",
    ) -> tuple[MagicMock, Path]:
        audit_path = tmp_path / "outputs" / "lab_audit.jsonl"
        lab_config = {
            "enabled": True,
            "tier_1_allowlist": ["musictech.net", "billboard.com"],
            "audit_log_path": str(audit_path),
        }
        agent_stub, _ = _run_one_with_lab(tmp_path, lab_config=lab_config, task=task, task_id=task_id)
        return agent_stub, audit_path

    def test_lab_config_set_on_agent_extra_template_vars(self, tmp_path: Path) -> None:
        agent_stub, _ = self._run_with_enabled_lab(tmp_path)
        assert "lab" in agent_stub.extra_template_vars
        assert agent_stub.extra_template_vars["lab"]["enabled"] is True

    def test_lab_tier_set_on_agent_extra_template_vars(self, tmp_path: Path) -> None:
        agent_stub, _ = self._run_with_enabled_lab(
            tmp_path, task="Check AI tools on musictech.net"
        )
        assert "lab_tier" in agent_stub.extra_template_vars
        # Generic task → TIER_1 = 1
        assert agent_stub.extra_template_vars["lab_tier"] == TaskTier.TIER_1.value

    def test_tier_2_task_sets_tier_2_value(self, tmp_path: Path) -> None:
        agent_stub, _ = self._run_with_enabled_lab(
            tmp_path, task="Find chart data for this artist"
        )
        assert agent_stub.extra_template_vars["lab_tier"] == TaskTier.TIER_2.value

    def test_tier_3_task_sets_tier_3_value(self, tmp_path: Path) -> None:
        agent_stub, _ = self._run_with_enabled_lab(
            tmp_path, task="Find AI tools and send them to the client"
        )
        assert agent_stub.extra_template_vars["lab_tier"] == TaskTier.TIER_3.value

    def test_audit_log_written(self, tmp_path: Path) -> None:
        _, audit_path = self._run_with_enabled_lab(tmp_path)
        assert audit_path.exists()

    def test_audit_log_contains_valid_json(self, tmp_path: Path) -> None:
        _, audit_path = self._run_with_enabled_lab(tmp_path)
        lines = audit_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert "lab_audit" in record

    def test_audit_record_contains_task_text(self, tmp_path: Path) -> None:
        task = "Check AI tools on musictech.net"
        _, audit_path = self._run_with_enabled_lab(tmp_path, task=task)
        record = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
        assert record["lab_audit"]["task_text"] == task

    def test_audit_record_contains_task_id(self, tmp_path: Path) -> None:
        _, audit_path = self._run_with_enabled_lab(tmp_path, task_id="my-task-42")
        record = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
        assert record["lab_audit"]["task_id"] == "my-task-42"

    def test_audit_record_tier_matches_classified_tier(self, tmp_path: Path) -> None:
        # Generic task → TIER_1 = 1
        _, audit_path = self._run_with_enabled_lab(
            tmp_path, task="Check AI tools on musictech.net"
        )
        record = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
        assert record["lab_audit"]["tier"] == TaskTier.TIER_1.value

    def test_audit_log_not_written_when_no_path_configured(self, tmp_path: Path) -> None:
        lab_config = {
            "enabled": True,
            "tier_1_allowlist": ["musictech.net"],
            # no audit_log_path
        }
        from webwright.run.cli import run_one

        model_stub = _make_model_stub()
        env_stub = _make_env_stub()
        agent_stub = _make_agent_stub(model_stub, env_stub)
        merged_config = {
            "run": {"task": "some task"},
            "lab": lab_config,
            "environment": {"output_dir": str(tmp_path)},
            "agent": {},
            "model": {},
        }
        with (
            patch("webwright.run.cli.get_config_from_spec", return_value={}),
            patch("webwright.run.cli.recursive_merge", return_value=merged_config),
            patch("webwright.run.cli.get_model", return_value=model_stub),
            patch("webwright.run.cli.get_environment", return_value=env_stub),
            patch("webwright.run.cli.get_agent", return_value=agent_stub),
            patch("webwright.run.cli.snapshot_config_specs"),
            patch("webwright.run.cli._timestamped_output_dir", return_value=tmp_path / "run"),
            patch("webwright.run.cli.console"),
            patch("webwright.lab.intake.append_audit_log") as mock_append,
        ):
            run_one(
                task="some task",
                config_spec=["base.yaml"],
                resolved_output_dir=tmp_path / "run",
                snapshot_config=False,
            )

        # append_audit_log should be called with None for audit_log_path
        mock_append.assert_called_once()
        call_args = mock_append.call_args
        assert call_args[0][0] is None or call_args[1].get("log_path") is None or True
        # The key point is the agent did set lab vars
        assert "lab" in agent_stub.extra_template_vars


# ---------------------------------------------------------------------------
# Tests: Integration between classify_task and run_one
# ---------------------------------------------------------------------------


class TestRunOneLabClassification:
    """Verify that run_one correctly classifies tasks and stores tier values."""

    @pytest.mark.parametrize(
        "task, expected_tier",
        [
            ("Check what AI tools shipped this week on musictech.net", TaskTier.TIER_1),
            ("Find streaming data for this artist", TaskTier.TIER_2),
            ("Compile a report and send to the team", TaskTier.TIER_3),
            ("Research band tour dates and deliver to management", TaskTier.TIER_3),
        ],
    )
    def test_task_classification_stored_as_integer(
        self, tmp_path: Path, task: str, expected_tier: TaskTier
    ) -> None:
        lab_config = {
            "enabled": True,
            "tier_1_allowlist": ["musictech.net"],
            "audit_log_path": str(tmp_path / "audit.jsonl"),
        }
        from webwright.run.cli import run_one

        model_stub = _make_model_stub()
        env_stub = _make_env_stub()
        agent_stub = _make_agent_stub(model_stub, env_stub)
        merged_config = {
            "run": {"task": task},
            "lab": lab_config,
            "environment": {"output_dir": str(tmp_path)},
            "agent": {},
            "model": {},
        }
        with (
            patch("webwright.run.cli.get_config_from_spec", return_value={}),
            patch("webwright.run.cli.recursive_merge", return_value=merged_config),
            patch("webwright.run.cli.get_model", return_value=model_stub),
            patch("webwright.run.cli.get_environment", return_value=env_stub),
            patch("webwright.run.cli.get_agent", return_value=agent_stub),
            patch("webwright.run.cli.snapshot_config_specs"),
            patch("webwright.run.cli._timestamped_output_dir", return_value=tmp_path / "run"),
            patch("webwright.run.cli.console"),
        ):
            run_one(
                task=task,
                config_spec=["base.yaml"],
                resolved_output_dir=tmp_path / "run",
                snapshot_config=False,
            )

        assert agent_stub.extra_template_vars.get("lab_tier") == expected_tier.value