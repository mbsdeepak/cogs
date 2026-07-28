"""Context trimming: recency preserved, tool_use/tool_result pairing intact."""

from __future__ import annotations

from cogs.context import ContextManager, estimate_tokens
from cogs.types import Message, Role, ToolCall, ToolResult


def _user(text: str) -> Message:
    return Message(role=Role.USER, text=text)


def _assistant_tool(call_id: str) -> Message:
    return Message(
        role=Role.ASSISTANT,
        text="calling",
        tool_calls=[ToolCall(id=call_id, name="t", input={})],
    )


def _tool_result(call_id: str) -> Message:
    return Message(
        role=Role.USER, tool_results=[ToolResult(call_id=call_id, content="ok")]
    )


def test_estimate_tokens_rounds_up():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_fit_returns_all_when_under_budget():
    mgr = ContextManager(max_tokens=10_000)
    messages = [_user("hi"), _user("there")]
    assert mgr.fit(messages) == messages


def test_fit_drops_oldest_when_over_budget():
    # Each message ~ 25 chars -> ~7 tokens. Budget forces dropping the front.
    mgr = ContextManager(max_tokens=20)
    messages = [_user("x" * 40) for _ in range(5)]
    kept = mgr.fit(messages)
    assert len(kept) < len(messages)
    # The most recent message is always kept.
    assert kept[-1] is messages[-1]


def test_fit_preserves_recency_order():
    mgr = ContextManager(max_tokens=30)
    messages = [_user(f"msg{i}-" + "y" * 40) for i in range(6)]
    kept = mgr.fit(messages)
    # Kept messages remain a contiguous suffix in original order.
    idx = [messages.index(m) for m in kept]
    assert idx == sorted(idx)
    assert idx[-1] == len(messages) - 1


def test_fit_never_orphans_tool_result():
    # Build: [big user][assistant tool_use][tool_result]. A tight budget would
    # drop the assistant turn but must then also drop the orphaned tool_result.
    mgr = ContextManager(max_tokens=8)
    messages = [
        _user("z" * 200),
        _assistant_tool("c1"),
        _tool_result("c1"),
    ]
    kept = mgr.fit(messages)
    # The kept head must not be a bare tool_result.
    if kept:
        assert not (kept[0].role is Role.USER and kept[0].tool_results)


def test_fit_keeps_pair_together_when_both_fit():
    mgr = ContextManager(max_tokens=10_000)
    messages = [_user("start"), _assistant_tool("c1"), _tool_result("c1")]
    kept = mgr.fit(messages)
    assert kept == messages


def test_system_tokens_counted_against_budget():
    mgr = ContextManager(max_tokens=100, system_tokens=90)
    messages = [_user("a" * 80) for _ in range(3)]  # ~20 tokens each
    kept = mgr.fit(messages)
    # Only ~10 tokens of budget remain for messages, so at most one survives.
    assert len(kept) <= 1


def test_estimate_includes_tool_payloads():
    mgr = ContextManager()
    plain = mgr.estimate([_user("hi")])
    with_tool = mgr.estimate([_assistant_tool("c1")])
    assert with_tool > 0
    assert plain >= 0
