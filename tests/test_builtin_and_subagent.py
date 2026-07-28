"""Builtin tools and sub-agent delegation."""

from __future__ import annotations

from cogs.agent import Agent
from cogs.tools import ToolRegistry
from cogs.tools_builtin import (
    default_registry,
    list_dir,
    read_file,
    run_bash,
    write_file,
)
from cogs.types import ToolCall

from .conftest import ScriptedProvider, text_turn, tool_turn


def test_read_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hi there", encoding="utf-8")
    assert read_file(str(f)) == "hi there"


def test_read_file_missing(tmp_path):
    from cogs.tools import ToolError

    try:
        read_file(str(tmp_path / "nope.txt"))
    except ToolError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ToolError")


def test_read_file_truncation(tmp_path, monkeypatch):
    import cogs.tools_builtin as tb

    monkeypatch.setattr(tb, "_MAX_READ_BYTES", 5)
    f = tmp_path / "big.txt"
    f.write_text("0123456789", encoding="utf-8")
    out = tb.read_file(str(f))
    assert "truncated" in out


def test_list_dir(tmp_path):
    (tmp_path / "a.txt").write_text("", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    out = list_dir(str(tmp_path))
    assert "a.txt" in out
    assert "sub/" in out


def test_write_file(tmp_path):
    dest = tmp_path / "nested" / "out.txt"
    msg = write_file(str(dest), "payload")
    assert dest.read_text(encoding="utf-8") == "payload"
    assert "7 bytes" in msg


def test_run_bash_allowlisted():
    out = run_bash("echo hello")
    assert "hello" in out


def test_run_bash_rejects_non_allowlisted():
    from cogs.tools import ToolError

    try:
        run_bash("rm -rf /")
    except ToolError as exc:
        assert "allowlist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ToolError")


def test_run_bash_rejects_metacharacters():
    from cogs.tools import ToolError

    try:
        run_bash("echo hi && rm foo")
    except ToolError as exc:
        assert "metacharacters" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ToolError")


def test_default_registry_has_all_builtins():
    reg = default_registry()
    for name in ("read_file", "list_dir", "write_file", "run_bash", "finish"):
        assert name in reg


def test_builtin_specs_have_descriptions():
    reg = default_registry()
    for spec in reg.specs():
        assert spec.description
        assert spec.input_schema["type"] == "object"


def test_subagent_delegation_returns_child_answer():
    # Parent asks to spawn a sub-agent; sub-agent produces an answer.
    child_registry = ToolRegistry()
    provider = ScriptedProvider(
        [
            # Parent turn 1: call spawn_agent.
            tool_turn(
                "delegating",
                ToolCall(
                    id="p1",
                    name="spawn_agent",
                    input={"task": "compute the answer"},
                ),
            ),
            # Child turn (uses next provider call): answer directly.
            text_turn("child says 42"),
            # Parent turn 2: final answer after receiving the child's result.
            text_turn("The sub-agent reported: child says 42"),
        ]
    )
    parent = Agent(provider, system="parent")
    parent.register_subagent_tool(registry=child_registry)
    answer = parent.run("delegate this")
    assert answer == "The sub-agent reported: child says 42"

    # The child's answer flowed back as the spawn_agent tool result.
    result_msg = next(m for m in parent.messages if m.tool_results)
    assert result_msg.tool_results[0].content == "child says 42"


def test_subagent_usage_rolls_up_into_parent():
    child_registry = ToolRegistry()
    provider = ScriptedProvider(
        [
            tool_turn(
                "delegating",
                ToolCall(id="p1", name="spawn_agent", input={"task": "x"}),
            ),
            text_turn("child done"),  # child: usage (10,5)
            text_turn("parent done"),  # parent final: usage (10,5)
        ]
    )
    parent = Agent(provider, system="parent")
    parent.register_subagent_tool(registry=child_registry)
    parent.run("go")
    # parent turn1 (20,8) + child (10,5) + parent turn2 (10,5) = (40,18)
    assert parent.total_usage.input_tokens == 40
    assert parent.total_usage.output_tokens == 18
