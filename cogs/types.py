"""Normalized data types spoken by the agent loop.

These dataclasses are the *lingua franca* of the runtime. Providers translate
their SDK-specific request/response shapes into these types, so the agent loop,
tracer, and context manager never import a provider SDK directly. This is what
makes providers swappable and makes deterministic record/replay possible: a
recorded ``AssistantTurn`` is just data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Conversation roles, matching the Messages API vocabulary."""

    USER = "user"
    ASSISTANT = "assistant"


class StopReason(StrEnum):
    """Why the model stopped generating.

    Mirrors the Messages API ``stop_reason`` field. ``REFUSAL`` and
    ``MAX_TOKENS`` are handled explicitly by the agent loop.
    """

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    STOP_SEQUENCE = "stop_sequence"


@dataclass(frozen=True)
class TextBlock:
    """A block of assistant text."""

    text: str


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to invoke a tool.

    ``id`` is the provider-assigned ``tool_use`` id; it MUST be echoed back in
    the matching :class:`ToolResult` so the provider can pair them.
    """

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The outcome of executing a tool, fed back to the model.

    ``call_id`` matches the originating :class:`ToolCall.id`. ``is_error`` marks
    failed executions (denied permissions, raised exceptions, bad input) so the
    model can recover rather than assuming success.
    """

    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Usage:
    """Token accounting for a single provider call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Message:
    """A single conversation turn in normalized form.

    The loop keeps a list of these. A user turn typically carries text or a
    batch of tool results; an assistant turn carries text and/or tool calls.
    Exactly one of the payload fields is populated per logical turn, but the
    dataclass permits mixed content to mirror the provider wire format.
    """

    role: Role
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass(frozen=True)
class AssistantTurn:
    """A normalized model response.

    This is the unit the loop reasons about and the tracer records. Replaying a
    recorded ``AssistantTurn`` reproduces a run exactly, with zero network I/O.

    ``reasoning`` holds a reasoning model's separate thinking output (e.g.
    Sarvam's ``reasoning_content``) when present. It is kept apart from ``text``
    so chain-of-thought never leaks into the user-facing answer; the agent loop
    ignores it, but tracing/observability can surface it. Empty for providers
    that do not return it.
    """

    text: str
    tool_calls: list[ToolCall]
    stop_reason: StopReason
    usage: Usage
    reasoning: str = ""


@dataclass(frozen=True)
class ToolSpec:
    """A provider-agnostic tool definition.

    ``input_schema`` is a JSON Schema object. :func:`cogs.tools.tool` generates
    these from function signatures; the provider serializes them into the
    ``tools`` array of a Messages API request.
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_api(self) -> dict[str, Any]:
        """Render as an Anthropic tool definition dict."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
