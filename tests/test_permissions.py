"""Permission gating: deny -> errored result; ask -> confirm callback."""

from __future__ import annotations

from cogs.agent import Agent
from cogs.permissions import Decision, PermissionPolicy, allow_all, gated
from cogs.tools import ToolRegistry, tool
from cogs.types import ToolCall

from .conftest import text_turn, tool_turn


def _registry_with_delete(record: list[str]) -> ToolRegistry:
    reg = ToolRegistry()

    @tool
    def delete_thing(name: str) -> str:
        """Delete a thing."""
        record.append(name)
        return f"deleted {name}"

    reg.register(delete_thing)
    return reg


def test_allow_all_permits():
    policy = allow_all()
    allowed, reason = policy.check(ToolCall(id="1", name="anything", input={}))
    assert allowed is True
    assert reason is None


def test_deny_blocks_with_reason():
    policy = PermissionPolicy(rules={"danger": Decision.DENY})
    allowed, reason = policy.check(ToolCall(id="1", name="danger", input={}))
    assert allowed is False
    assert "denied" in reason


def test_ask_defers_to_confirm_true():
    calls = []
    policy = PermissionPolicy(
        rules={"x": Decision.ASK}, confirm=lambda c: calls.append(c) or True
    )
    allowed, reason = policy.check(ToolCall(id="1", name="x", input={}))
    assert allowed is True
    assert len(calls) == 1


def test_ask_defers_to_confirm_false():
    policy = PermissionPolicy(rules={"x": Decision.ASK}, confirm=lambda c: False)
    allowed, reason = policy.check(ToolCall(id="1", name="x", input={}))
    assert allowed is False
    assert "declined" in reason


def test_ask_default_confirm_fails_closed():
    # No confirm callback supplied -> ASK behaves like DENY.
    policy = PermissionPolicy(rules={"x": Decision.ASK})
    allowed, _ = policy.check(ToolCall(id="1", name="x", input={}))
    assert allowed is False


def test_denied_tool_produces_errored_result_and_never_runs(scripted):
    record: list[str] = []
    provider = scripted(
        tool_turn(
            "deleting",
            ToolCall(id="t1", name="delete_thing", input={"name": "prod-db"}),
        ),
        text_turn("I could not delete it."),
    )
    policy = PermissionPolicy(default=Decision.DENY)
    agent = Agent(
        provider,
        system="sys",
        registry=_registry_with_delete(record),
        permissions=policy,
    )
    agent.run("delete the prod db")

    # The tool never executed.
    assert record == []
    # The model received an errored tool_result.
    result_msg = next(m for m in agent.messages if m.tool_results)
    result = result_msg.tool_results[0]
    assert result.is_error is True
    assert "denied" in result.content


def test_ask_confirm_true_lets_tool_run(scripted):
    record: list[str] = []
    provider = scripted(
        tool_turn(
            "deleting",
            ToolCall(id="t1", name="delete_thing", input={"name": "tmp"}),
        ),
        text_turn("done"),
    )
    policy = PermissionPolicy(
        rules={"delete_thing": Decision.ASK}, confirm=lambda c: True
    )
    agent = Agent(
        provider,
        system="sys",
        registry=_registry_with_delete(record),
        permissions=policy,
    )
    agent.run("delete tmp")
    assert record == ["tmp"]
    result_msg = next(m for m in agent.messages if m.tool_results)
    assert result_msg.tool_results[0].is_error is False


def test_gated_policy_defaults_side_effecting_tools_to_ask():
    policy = gated()
    assert policy.decision_for("run_bash") is Decision.ASK
    assert policy.decision_for("write_file") is Decision.ASK
    assert policy.decision_for("read_file") is Decision.ALLOW
