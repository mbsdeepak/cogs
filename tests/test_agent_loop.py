"""The full agent loop: message assembly, tool_result pairing, termination."""

from __future__ import annotations

import pytest

from cogs.agent import Agent, AgentError
from cogs.tools import ToolRegistry, tool
from cogs.types import Role, StopReason, ToolCall, Usage

from .conftest import text_turn, tool_turn


def _registry_with_add() -> ToolRegistry:
    reg = ToolRegistry()

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    reg.register(add)
    return reg


def test_single_turn_no_tools(scripted):
    provider = scripted(text_turn("The answer is 42."))
    agent = Agent(provider, system="sys")
    assert agent.run("what is the answer?") == "The answer is 42."


def test_tool_use_loop_assembles_messages(scripted):
    provider = scripted(
        tool_turn("computing", ToolCall(id="t1", name="add", input={"a": 2, "b": 3})),
        text_turn("The sum is 5."),
    )
    agent = Agent(provider, system="sys", registry=_registry_with_add())
    answer = agent.run("add 2 and 3")
    assert answer == "The sum is 5."

    # Message history: user, assistant(tool_use), user(tool_result), assistant(text)
    roles = [m.role for m in agent.messages]
    assert roles == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]

    assistant_call = agent.messages[1]
    assert assistant_call.tool_calls[0].name == "add"

    tool_result_msg = agent.messages[2]
    assert tool_result_msg.role is Role.USER
    assert len(tool_result_msg.tool_results) == 1
    assert tool_result_msg.tool_results[0].call_id == "t1"
    assert tool_result_msg.tool_results[0].content == "5"


def test_all_tool_results_in_one_user_message(scripted):
    provider = scripted(
        tool_turn(
            "parallel",
            ToolCall(id="t1", name="add", input={"a": 1, "b": 1}),
            ToolCall(id="t2", name="add", input={"a": 10, "b": 5}),
        ),
        text_turn("done"),
    )
    agent = Agent(provider, system="sys", registry=_registry_with_add())
    agent.run("add stuff")

    # Exactly one user message carries BOTH tool results, correctly paired.
    result_msgs = [m for m in agent.messages if m.tool_results]
    assert len(result_msgs) == 1
    results = result_msgs[0].tool_results
    assert [r.call_id for r in results] == ["t1", "t2"]
    assert [r.content for r in results] == ["2", "15"]


def test_second_provider_call_sees_prior_history(scripted):
    provider = scripted(
        tool_turn("t", ToolCall(id="t1", name="add", input={"a": 1, "b": 2})),
        text_turn("3"),
    )
    agent = Agent(provider, system="sys", registry=_registry_with_add())
    agent.run("go")

    # The provider's second call should have received the assistant tool_use
    # turn and the tool_result user turn in its messages.
    second_call_messages = provider.calls[1]["messages"]
    assert any(m.tool_calls for m in second_call_messages)
    assert any(m.tool_results for m in second_call_messages)


def test_max_tokens_stops_gracefully(scripted):
    from cogs.types import AssistantTurn

    turn = AssistantTurn(
        text="partial answer",
        tool_calls=[],
        stop_reason=StopReason.MAX_TOKENS,
        usage=Usage(5, 5),
    )
    provider = scripted(turn)
    agent = Agent(provider, system="sys")
    answer = agent.run("write a novel")
    assert answer == "partial answer"


def test_refusal_stops_gracefully(scripted):
    from cogs.types import AssistantTurn

    turn = AssistantTurn(
        text="I can't help with that.",
        tool_calls=[],
        stop_reason=StopReason.REFUSAL,
        usage=Usage(3, 3),
    )
    provider = scripted(turn)
    agent = Agent(provider, system="sys")
    assert agent.run("do something bad") == "I can't help with that."


def test_step_budget_exhausted_raises(scripted):
    # Provider always asks for a tool -> never terminates.
    turns = [
        tool_turn("loop", ToolCall(id=f"t{i}", name="add", input={"a": 1, "b": 1}))
        for i in range(10)
    ]
    provider = scripted(*turns)
    agent = Agent(provider, system="sys", registry=_registry_with_add(), max_steps=3)
    with pytest.raises(AgentError):
        agent.run("loop forever")


def test_usage_accumulates_across_turns(scripted):
    provider = scripted(
        tool_turn("t", ToolCall(id="t1", name="add", input={"a": 1, "b": 1})),
        text_turn("2"),
    )
    agent = Agent(provider, system="sys", registry=_registry_with_add())
    agent.run("go")
    # tool_turn usage (20,8) + text_turn usage (10,5)
    assert agent.total_usage.input_tokens == 30
    assert agent.total_usage.output_tokens == 13


def test_multi_turn_session_preserves_history(scripted):
    provider = scripted(text_turn("hi Alice"), text_turn("your name is Alice"))
    agent = Agent(provider, system="sys")
    agent.run("my name is Alice")
    agent.run("what is my name?")
    # Both user turns and both assistant turns are retained.
    user_texts = [m.text for m in agent.messages if m.role is Role.USER]
    assert user_texts == ["my name is Alice", "what is my name?"]
