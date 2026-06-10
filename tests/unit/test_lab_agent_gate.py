"""Tests for the LAB domain allowlist gate added to DefaultAgent.execute_actions()."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from webwright.agents.default import DefaultAgent


# ---------------------------------------------------------------------------
# Minimal stubs for Model and Environment
# ---------------------------------------------------------------------------


def _make_model() -> MagicMock:
    """Return a MagicMock that mimics the Model protocol."""
    model = MagicMock()
    model.config = {}
    model.format_message.side_effect = lambda **kw: {
        "role": kw.get("role", "user"),
        "content": kw.get("content", ""),
        "extra": kw.get("extra", {}),
    }
    model.format_observation_messages.return_value = [
        {"role": "user", "content": "obs", "extra": {"observation": {}}}
    ]
    model.get_template_vars.return_value = {}
    model.serialize.return_value = {"model": {}}
    return model


def _make_env() -> MagicMock:
    """Return a MagicMock that mimics the Environment protocol."""
    env = MagicMock()
    env.config = {}
    env.execute.return_value = {"observation": {"success": True}}
    env.get_template_vars.return_value = {}
    env.serialize.return_value = {"environment": {}}
    return env


def _make_agent(extra_template_vars: dict[str, Any] | None = None) -> DefaultAgent:
    """Create a DefaultAgent with minimal config and optional extra_template_vars."""
    model = _make_model()
    env = _make_env()
    agent = DefaultAgent(
        model=model,
        env=env,
        system_template="system",
        instance_template="instance",
        debug_log=False,
    )
    if extra_template_vars:
        agent.extra_template_vars.update(extra_template_vars)
    return agent


def _make_message(actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a minimal assistant message with actions but done=False."""
    return {
        "role": "assistant",
        "content": "thinking",
        "extra": {
            "done": False,
            "actions": actions,
        },
    }


# ---------------------------------------------------------------------------
# Tests: LAB gate disabled (no lab config)
# ---------------------------------------------------------------------------


class TestLabGateDisabled:
    def test_no_lab_config_executes_actions(self) -> None:
        agent = _make_agent()
        action = {"python_code": 'page.goto("https://evil.com")'}
        message = _make_message([action])

        with patch.object(agent, "_write_debug_step_artifact"):
            agent.execute_actions(message)

        # Environment.execute should be called since no gate is in place
        agent.env.execute.assert_called_once_with(action)

    def test_lab_config_disabled_flag_executes_actions(self) -> None:
        agent = _make_agent(extra_template_vars={"lab": {"enabled": False, "tier_1_allowlist": ["allowed.com"]}})
        action = {"python_code": 'page.goto("https://evil.com")'}
        message = _make_message([action])

        with patch.object(agent, "_write_debug_step_artifact"):
            agent.execute_actions(message)

        agent.env.execute.assert_called_once_with(action)


# ---------------------------------------------------------------------------
# Tests: LAB gate enabled, clean code
# ---------------------------------------------------------------------------


class TestLabGateEnabledClean:
    def test_allowed_domain_in_python_code_passes(self) -> None:
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": ["musictech.net"],
                }
            }
        )
        action = {"python_code": 'page.goto("https://musictech.net/news")'}
        message = _make_message([action])

        with patch.object(agent, "_write_debug_step_artifact"):
            agent.execute_actions(message)

        agent.env.execute.assert_called_once_with(action)

    def test_no_goto_in_code_passes(self) -> None:
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": ["musictech.net"],
                }
            }
        )
        action = {"python_code": "x = 1 + 1\nprint(x)"}
        message = _make_message([action])

        with patch.object(agent, "_write_debug_step_artifact"):
            agent.execute_actions(message)

        agent.env.execute.assert_called_once_with(action)

    def test_empty_allowlist_skips_check(self) -> None:
        # Empty allowlist means check_domain_allowlist returns None (no-op)
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": [],
                }
            }
        )
        action = {"python_code": 'page.goto("https://any.com")'}
        message = _make_message([action])

        with patch.object(agent, "_write_debug_step_artifact"):
            agent.execute_actions(message)

        agent.env.execute.assert_called_once_with(action)

    def test_allowed_domain_in_bash_command_passes(self) -> None:
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": ["billboard.com"],
                }
            }
        )
        action = {"bash_command": 'goto("https://billboard.com/charts")'}
        message = _make_message([action])

        with patch.object(agent, "_write_debug_step_artifact"):
            agent.execute_actions(message)

        agent.env.execute.assert_called_once_with(action)

    def test_multiple_allowed_actions_all_execute(self) -> None:
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": ["musictech.net", "billboard.com"],
                }
            }
        )
        actions = [
            {"python_code": 'page.goto("https://musictech.net")'},
            {"python_code": 'page.goto("https://billboard.com")'},
        ]
        message = _make_message(actions)

        with patch.object(agent, "_write_debug_step_artifact"):
            agent.execute_actions(message)

        assert agent.env.execute.call_count == 2


# ---------------------------------------------------------------------------
# Tests: LAB gate enabled, violation detected
# ---------------------------------------------------------------------------


class TestLabGateEnabledViolation:
    def _run_with_violation(self, code: str, allowlist: list[str]) -> list[dict[str, Any]]:
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": allowlist,
                }
            }
        )
        action = {"python_code": code}
        message = _make_message([action])

        with patch.object(agent, "_write_debug_step_artifact"):
            return agent.execute_actions(message)

    def test_blocked_domain_prevents_execution(self) -> None:
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": ["musictech.net"],
                }
            }
        )
        action = {"python_code": 'page.goto("https://evil.com/steal")'}
        message = _make_message([action])

        with patch.object(agent, "_write_debug_step_artifact"):
            agent.execute_actions(message)

        # Environment.execute must NOT be called
        agent.env.execute.assert_not_called()

    def test_blocked_domain_adds_lab_intake_gate_message(self) -> None:
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": ["musictech.net"],
                }
            }
        )
        action = {"python_code": 'page.goto("https://evil.com/steal")'}
        message = _make_message([action])

        with patch.object(agent, "_write_debug_step_artifact"):
            added = agent.execute_actions(message)

        assert len(added) == 1
        interrupt_type = added[0].get("extra", {}).get("interrupt_type")
        assert interrupt_type == "LABIntakeGate"

    def test_blocked_message_content_mentions_violation(self) -> None:
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": ["musictech.net"],
                }
            }
        )
        action = {"python_code": 'page.goto("https://evil.com")'}
        message = _make_message([action])

        with patch.object(agent, "_write_debug_step_artifact"):
            added = agent.execute_actions(message)

        content = added[0].get("content", "")
        assert "LAB intake gate blocked" in content

    def test_blocked_message_instructs_revision(self) -> None:
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": ["musictech.net"],
                }
            }
        )
        action = {"python_code": 'page.goto("https://evil.com")'}
        message = _make_message([action])

        with patch.object(agent, "_write_debug_step_artifact"):
            added = agent.execute_actions(message)

        content = added[0].get("content", "")
        assert "LAB-approved domains" in content

    def test_bash_command_violation_also_blocked(self) -> None:
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": ["musictech.net"],
                }
            }
        )
        action = {"bash_command": 'goto("https://notallowed.org")'}
        message = _make_message([action])

        with patch.object(agent, "_write_debug_step_artifact"):
            agent.execute_actions(message)

        agent.env.execute.assert_not_called()

    def test_python_code_takes_precedence_over_bash_command(self) -> None:
        """If python_code is set, it is checked; bash_command is ignored per `or` logic."""
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": ["musictech.net"],
                }
            }
        )
        # python_code has an evil domain; bash_command has an allowed one
        action = {
            "python_code": 'page.goto("https://evil.com")',
            "bash_command": 'goto("https://musictech.net")',
        }
        message = _make_message([action])

        with patch.object(agent, "_write_debug_step_artifact"):
            agent.execute_actions(message)

        # python_code is picked first and should trigger a block
        agent.env.execute.assert_not_called()

    def test_first_violating_action_stops_all_execution(self) -> None:
        """If the first action violates, no subsequent actions are executed."""
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": ["musictech.net"],
                }
            }
        )
        actions = [
            {"python_code": 'page.goto("https://evil.com")'},
            {"python_code": 'page.goto("https://musictech.net")'},
        ]
        message = _make_message(actions)

        with patch.object(agent, "_write_debug_step_artifact"):
            agent.execute_actions(message)

        agent.env.execute.assert_not_called()

    def test_violation_message_is_added_to_agent_messages(self) -> None:
        """The returned gate message should also appear in agent.messages."""
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": ["musictech.net"],
                }
            }
        )
        action = {"python_code": 'page.goto("https://evil.com")'}
        message = _make_message([action])

        with patch.object(agent, "_write_debug_step_artifact"):
            agent.execute_actions(message)

        # At least one message with LABIntakeGate should now be in agent.messages
        gate_messages = [
            m for m in agent.messages
            if m.get("extra", {}).get("interrupt_type") == "LABIntakeGate"
        ]
        assert len(gate_messages) == 1

    def test_no_actions_with_lab_enabled_executes_nothing(self) -> None:
        agent = _make_agent(
            extra_template_vars={
                "lab": {
                    "enabled": True,
                    "tier_1_allowlist": ["musictech.net"],
                }
            }
        )
        message = _make_message([])

        with patch.object(agent, "_write_debug_step_artifact"):
            agent.execute_actions(message)

        agent.env.execute.assert_not_called()