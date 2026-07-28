"""Tracing + record/replay round-trip determinism."""

from __future__ import annotations

from cogs.agent import Agent
from cogs.tools import ToolRegistry, tool
from cogs.trace import RecordingProvider, ReplayProvider, Tracer, load_turns
from cogs.types import AssistantTurn, StopReason, ToolCall, Usage

from .conftest import ScriptedProvider, text_turn, tool_turn


def _registry() -> ToolRegistry:
    reg = ToolRegistry()

    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    reg.register(add)
    return reg


def test_replay_returns_turns_in_order():
    turns = [
        tool_turn("t", ToolCall(id="1", name="add", input={"a": 1, "b": 1})),
        text_turn("done"),
    ]
    provider = ReplayProvider(turns)
    assert provider.complete(None, [], []) is turns[0]
    assert provider.complete(None, [], []) is turns[1]


def test_replay_exhaustion_raises():
    provider = ReplayProvider([text_turn("only one")])
    provider.complete(None, [], [])
    try:
        provider.complete(None, [], [])
    except IndexError as exc:
        assert "exhausted" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected IndexError")


def test_tracer_records_provider_calls():
    with Tracer() as tracer:
        tracer.record_provider_call(0, text_turn("hi"))
    events = [e for e in tracer.events if e["event"] == "provider_call"]
    assert len(events) == 1
    assert events[0]["turn"]["text"] == "hi"


def test_tracer_records_tool_executions():
    from cogs.types import ToolResult

    with Tracer() as tracer:
        tracer.record_tool_execution(
            ToolCall(id="1", name="add", input={"a": 1, "b": 2}),
            ToolResult(call_id="1", content="3"),
        )
    events = [e for e in tracer.events if e["event"] == "tool_execution"]
    assert events[0]["name"] == "add"
    assert events[0]["result"]["content"] == "3"


def test_cassette_write_and_load_roundtrip(tmp_path):
    path = tmp_path / "c.jsonl"
    original = [
        tool_turn("t", ToolCall(id="1", name="add", input={"a": 4, "b": 5})),
        text_turn("9"),
    ]
    with Tracer(path) as tracer:
        for i, turn in enumerate(original):
            tracer.record_provider_call(i, turn)

    loaded = load_turns(path)
    assert len(loaded) == 2
    assert loaded[0].tool_calls[0].input == {"a": 4, "b": 5}
    assert loaded[1].text == "9"
    assert loaded[0].stop_reason is StopReason.TOOL_USE


def test_record_then_replay_produces_identical_run(tmp_path):
    """The core determinism guarantee: record a run, replay it, get the same result."""
    cassette = tmp_path / "run.jsonl"

    # 1. Live run against a scripted provider, wrapped in a RecordingProvider.
    scripted = ScriptedProvider(
        [
            tool_turn("t", ToolCall(id="t1", name="add", input={"a": 7, "b": 8})),
            text_turn("The result is 15."),
        ]
    )
    with Tracer(cassette) as rec_tracer:
        recording = RecordingProvider(scripted, rec_tracer)
        live_agent = Agent(recording, system="sys", registry=_registry())
        live_answer = live_agent.run("add 7 and 8")

    # 2. Replay run from the recorded cassette.
    with Tracer() as replay_tracer:
        replay = ReplayProvider.from_cassette(cassette)
        replay_agent = Agent(replay, system="sys", registry=_registry())
        replay_answer = replay_agent.run("add 7 and 8")

    # 3. Identical final answers, identical tool executions, identical usage.
    assert live_answer == replay_answer == "The result is 15."

    def tool_events(tracer):
        return [
            (e["name"], e["input"], e["result"]["content"])
            for e in tracer.events
            if e["event"] == "tool_execution"
        ]

    assert tool_events(rec_tracer) == tool_events(replay_tracer)
    assert live_agent.total_usage == replay_agent.total_usage


def test_usage_serialization_survives_roundtrip(tmp_path):
    path = tmp_path / "c.jsonl"
    turn = AssistantTurn(
        text="hi",
        tool_calls=[],
        stop_reason=StopReason.END_TURN,
        usage=Usage(input_tokens=11, output_tokens=22, cache_read_tokens=33),
    )
    with Tracer(path) as tracer:
        tracer.record_provider_call(0, turn)
    loaded = load_turns(path)[0]
    assert loaded.usage.input_tokens == 11
    assert loaded.usage.output_tokens == 22
    assert loaded.usage.cache_read_tokens == 33
