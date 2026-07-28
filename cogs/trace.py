"""Structured tracing + deterministic record/replay.

Every provider call and tool execution is appended to a JSONL "cassette" as a
self-describing event. A recorded run can be replayed with zero network calls by
feeding the recorded :class:`~cogs.types.AssistantTurn`\\ s back through a
:class:`ReplayProvider`. This is the backbone of the test suite and of the
credential-free example.

A cassette line looks like::

    {"event": "provider_call", "index": 0, "turn": {...}}
    {"event": "tool_execution", "name": "read_file", "result": {...}}

Only ``provider_call`` events participate in replay; ``tool_execution`` events
are diagnostic (they let you inspect what the harness did without re-running
the tools).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .provider import Provider
from .types import (
    AssistantTurn,
    Message,
    StopReason,
    ToolCall,
    ToolResult,
    Usage,
)


def _turn_to_dict(turn: AssistantTurn) -> dict[str, Any]:
    return {
        "text": turn.text,
        "tool_calls": [asdict(c) for c in turn.tool_calls],
        "stop_reason": turn.stop_reason.value,
        "usage": asdict(turn.usage),
    }


def _turn_from_dict(data: dict[str, Any]) -> AssistantTurn:
    return AssistantTurn(
        text=data["text"],
        tool_calls=[
            ToolCall(id=c["id"], name=c["name"], input=c["input"])
            for c in data["tool_calls"]
        ],
        stop_reason=StopReason(data["stop_reason"]),
        usage=Usage(**data["usage"]),
    )


class Tracer:
    """Appends trace events to an in-memory list and, optionally, a JSONL file.

    Use as a context manager to guarantee the file handle is closed::

        with Tracer("run.jsonl") as tracer:
            ...
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.events: list[dict[str, Any]] = []
        self._fh = None
        if self.path is not None:
            self._fh = self.path.open("w", encoding="utf-8")

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        if self._fh is not None:
            self._fh.write(json.dumps(event) + "\n")
            self._fh.flush()

    def record_provider_call(self, index: int, turn: AssistantTurn) -> None:
        """Record a model completion (a replayable event)."""
        self._emit({"event": "provider_call", "index": index, "turn": _turn_to_dict(turn)})

    def record_tool_execution(self, call: ToolCall, result: ToolResult) -> None:
        """Record a tool execution (diagnostic, not replayed)."""
        self._emit(
            {
                "event": "tool_execution",
                "name": call.name,
                "input": call.input,
                "result": asdict(result),
            }
        )

    def record_event(self, name: str, **fields: Any) -> None:
        """Record an arbitrary diagnostic event."""
        self._emit({"event": name, **fields})


def load_turns(path: str | Path) -> list[AssistantTurn]:
    """Load the ordered ``provider_call`` turns from a cassette file."""
    turns: list[AssistantTurn] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        if event.get("event") == "provider_call":
            turns.append(_turn_from_dict(event["turn"]))
    return turns


class ReplayProvider:
    """A :class:`~cogs.provider.Provider` that replays recorded turns in order.

    Each call to :meth:`complete` returns the next recorded
    :class:`~cogs.types.AssistantTurn`, ignoring its arguments. This makes the
    entire agent loop runnable and testable offline: the recorded run drives the
    same tool executions and message assembly, deterministically.
    """

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self._turns = list(turns)
        self._index = 0

    @classmethod
    def from_cassette(cls, path: str | Path) -> ReplayProvider:
        """Build a replay provider from a JSONL cassette file."""
        return cls(load_turns(path))

    def complete(
        self,
        system: str | None,
        messages: list[Message],
        tools: Any,
    ) -> AssistantTurn:
        if self._index >= len(self._turns):
            raise IndexError(
                "ReplayProvider exhausted: the cassette has no more recorded "
                "turns. The live run made more provider calls than were "
                "recorded."
            )
        turn = self._turns[self._index]
        self._index += 1
        return turn

    @property
    def remaining(self) -> int:
        return len(self._turns) - self._index


class RecordingProvider:
    """Wraps a live :class:`~cogs.provider.Provider` and records each call.

    Every completion is written to ``tracer`` as a replayable ``provider_call``
    event, then returned unchanged. A cassette produced this way replays exactly
    via :class:`ReplayProvider`, giving record→replay round-trip determinism.
    """

    def __init__(self, inner: Provider, tracer: Tracer) -> None:
        self._inner = inner
        self._tracer = tracer
        self._index = 0

    def complete(
        self,
        system: str | None,
        messages: list[Message],
        tools: Any,
    ) -> AssistantTurn:
        turn = self._inner.complete(system, messages, tools)
        self._tracer.record_provider_call(self._index, turn)
        self._index += 1
        return turn


__all__ = [
    "Tracer",
    "ReplayProvider",
    "RecordingProvider",
    "load_turns",
]
