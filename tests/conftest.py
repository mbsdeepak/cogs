"""Shared test fixtures: a scripted stub provider for driving the agent loop.

Nothing here touches the network. :class:`ScriptedProvider` returns a
predetermined sequence of :class:`~cogs.types.AssistantTurn`\\ s and records the
arguments it was called with, so tests can assert on message assembly.
"""

from __future__ import annotations

import pytest

from cogs.provider import Provider
from cogs.types import AssistantTurn, StopReason, ToolCall, Usage


def text_turn(text: str) -> AssistantTurn:
    return AssistantTurn(
        text=text, tool_calls=[], stop_reason=StopReason.END_TURN, usage=Usage(10, 5)
    )


def tool_turn(text: str, *calls: ToolCall) -> AssistantTurn:
    return AssistantTurn(
        text=text,
        tool_calls=list(calls),
        stop_reason=StopReason.TOOL_USE,
        usage=Usage(20, 8),
    )


class ScriptedProvider(Provider):
    """A provider that returns a fixed list of turns and records call args."""

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self._turns = list(turns)
        self._index = 0
        self.calls: list[dict] = []

    def complete(self, system, messages, tools):  # type: ignore[override]
        self.calls.append(
            {"system": system, "messages": list(messages), "tools": list(tools)}
        )
        turn = self._turns[self._index]
        self._index += 1
        return turn


@pytest.fixture
def scripted():
    """Factory: build a ScriptedProvider from turns."""

    def _make(*turns: AssistantTurn) -> ScriptedProvider:
        return ScriptedProvider(list(turns))

    return _make
