"""Sarvam provider translation, offline. No SDK, no network.

We inject a ``transport`` stub that captures the JSON payload we would POST and
returns a canned OpenAI-shaped response, then assert on both the request we
build and the normalized turn we decode. This is the offline half of the
Sarvam tool-calling smoke test; the live half lives in
``examples/sarvam_smoke.py``.
"""

from __future__ import annotations

import json

from cogs.provider import SarvamProvider
from cogs.types import Message, Role, StopReason, ToolCall, ToolResult, ToolSpec


class FakeTransport:
    """Captures the posted payload and returns a canned response dict."""

    def __init__(self, response: dict) -> None:
        self._response = response
        self.captured: dict | None = None

    def __call__(self, payload: dict) -> dict:
        self.captured = payload
        return self._response


def _response(message: dict, finish_reason: str = "stop", usage: dict | None = None):
    return {
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20},
    }


def _provider(response: dict) -> tuple[SarvamProvider, FakeTransport]:
    transport = FakeTransport(response)
    return SarvamProvider(transport, api_key="test-key"), transport


# -- decode -----------------------------------------------------------------


def test_decode_text_response():
    provider, _ = _provider(_response({"role": "assistant", "content": "hello"}))
    turn = provider.complete("sys", [Message(role=Role.USER, text="hi")], [])
    assert turn.text == "hello"
    assert turn.stop_reason is StopReason.END_TURN
    assert turn.tool_calls == []
    assert turn.usage.input_tokens == 100
    assert turn.usage.output_tokens == 20


def test_decode_tool_call_parses_json_string_arguments():
    provider, _ = _provider(
        _response(
            {
                "role": "assistant",
                "content": "let me check",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        # Sarvam returns arguments as a JSON *string*.
                        "function": {"name": "get_x", "arguments": '{"q": "v"}'},
                    }
                ],
            },
            finish_reason="tool_calls",
        )
    )
    turn = provider.complete(None, [Message(role=Role.USER, text="hi")], [])
    assert turn.stop_reason is StopReason.TOOL_USE
    assert turn.text == "let me check"
    assert turn.tool_calls == [ToolCall(id="call_1", name="get_x", input={"q": "v"})]


def test_decode_tool_call_with_empty_arguments():
    provider, _ = _provider(
        _response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "now", "arguments": ""}}
                ],
            },
            finish_reason="tool_calls",
        )
    )
    turn = provider.complete(None, [Message(role=Role.USER, text="hi")], [])
    assert turn.tool_calls == [ToolCall(id="c1", name="now", input={})]
    assert turn.text == ""


def test_decode_malformed_arguments_do_not_crash():
    provider, _ = _provider(
        _response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{not json"}}
                ],
            },
            finish_reason="tool_calls",
        )
    )
    turn = provider.complete(None, [Message(role=Role.USER, text="hi")], [])
    assert turn.tool_calls == [ToolCall(id="c1", name="f", input={})]


def test_finish_reason_mapping():
    for wire, expected in [
        ("stop", StopReason.END_TURN),
        ("tool_calls", StopReason.TOOL_USE),
        ("function_call", StopReason.TOOL_USE),
        ("length", StopReason.MAX_TOKENS),
        ("content_filter", StopReason.REFUSAL),
        ("something_new", StopReason.END_TURN),  # unknown -> safe default
    ]:
        provider, _ = _provider(
            _response({"role": "assistant", "content": "x"}, finish_reason=wire)
        )
        turn = provider.complete(None, [Message(role=Role.USER, text="hi")], [])
        assert turn.stop_reason is expected, wire


# -- encode -----------------------------------------------------------------


def test_encode_system_prompt_becomes_first_message():
    provider, transport = _provider(_response({"role": "assistant", "content": "ok"}))
    provider.complete("system prompt", [Message(role=Role.USER, text="hi")], [])
    messages = transport.captured["messages"]
    assert messages[0] == {"role": "system", "content": "system prompt"}
    assert messages[1] == {"role": "user", "content": "hi"}
    assert transport.captured["model"] == "sarvam-105b"
    assert transport.captured["max_tokens"] == 4096


def test_encode_no_system_prompt_omits_system_message():
    provider, transport = _provider(_response({"role": "assistant", "content": "ok"}))
    provider.complete(None, [Message(role=Role.USER, text="hi")], [])
    assert transport.captured["messages"] == [{"role": "user", "content": "hi"}]


def test_encode_tool_results_split_into_separate_tool_messages():
    provider, transport = _provider(_response({"role": "assistant", "content": "ok"}))
    msg = Message(
        role=Role.USER,
        tool_results=[
            ToolResult(call_id="t1", content="42"),
            ToolResult(call_id="t2", content="boom", is_error=True),
        ],
    )
    provider.complete(None, [msg], [])
    assert transport.captured["messages"] == [
        {"role": "tool", "tool_call_id": "t1", "content": "42"},
        {"role": "tool", "tool_call_id": "t2", "content": "boom"},
    ]


def test_encode_assistant_tool_call_serializes_arguments_as_json_string():
    provider, transport = _provider(_response({"role": "assistant", "content": "ok"}))
    msg = Message(
        role=Role.ASSISTANT,
        text="calling",
        tool_calls=[ToolCall(id="t1", name="add", input={"a": 1})],
    )
    provider.complete(None, [msg], [])
    sent = transport.captured["messages"][0]
    assert sent["role"] == "assistant"
    assert sent["content"] == "calling"
    call = sent["tool_calls"][0]
    assert call == {
        "id": "t1",
        "type": "function",
        "function": {"name": "add", "arguments": json.dumps({"a": 1})},
    }


def test_encode_tools_use_openai_function_shape():
    provider, transport = _provider(_response({"role": "assistant", "content": "ok"}))
    spec = ToolSpec(
        name="add",
        description="Add.",
        input_schema={"type": "object", "properties": {}},
    )
    provider.complete(None, [Message(role=Role.USER, text="hi")], [spec])
    assert transport.captured["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_explicit_model_and_max_tokens_override():
    provider = SarvamProvider(
        FakeTransport(_response({"role": "assistant", "content": "ok"})),
        api_key="k",
        model="sarvam-30b",
        max_tokens=512,
    )
    assert provider.model == "sarvam-30b"
    assert provider.max_tokens == 512


def test_round_trip_survives_tool_loop_shape():
    """An assistant tool-call turn re-encodes into a valid follow-up request."""
    provider, transport = _provider(_response({"role": "assistant", "content": "done"}))
    history = [
        Message(role=Role.USER, text="add 1 and 2"),
        Message(
            role=Role.ASSISTANT,
            text=None,
            tool_calls=[ToolCall(id="t1", name="add", input={"a": 1, "b": 2})],
        ),
        Message(role=Role.USER, tool_results=[ToolResult(call_id="t1", content="3")]),
    ]
    provider.complete("sys", history, [])
    roles = [m["role"] for m in transport.captured["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]


# -- reasoning-model output -------------------------------------------------


def test_reasoning_content_captured_separately_from_text():
    provider, _ = _provider(
        _response(
            {
                "role": "assistant",
                "content": "The answer is 4.",
                "reasoning_content": "2 + 2, carry nothing, equals 4.",
            }
        )
    )
    turn = provider.complete(None, [Message(role=Role.USER, text="2+2?")], [])
    assert turn.text == "The answer is 4."
    # Chain-of-thought stays out of the user-facing answer.
    assert turn.reasoning == "2 + 2, carry nothing, equals 4."


def test_reasoning_alias_field_supported():
    provider, _ = _provider(
        _response({"role": "assistant", "content": "ok", "reasoning": "because"})
    )
    turn = provider.complete(None, [Message(role=Role.USER, text="hi")], [])
    assert turn.reasoning == "because"


def test_no_reasoning_field_defaults_empty():
    provider, _ = _provider(_response({"role": "assistant", "content": "ok"}))
    turn = provider.complete(None, [Message(role=Role.USER, text="hi")], [])
    assert turn.reasoning == ""


def test_reasoning_survives_trace_round_trip():
    from cogs.trace import _turn_from_dict, _turn_to_dict

    provider, _ = _provider(
        _response({"role": "assistant", "content": "a", "reasoning_content": "why"})
    )
    turn = provider.complete(None, [Message(role=Role.USER, text="hi")], [])
    restored = _turn_from_dict(_turn_to_dict(turn))
    assert restored.reasoning == "why"
    assert restored.text == "a"


# -- request-side options ---------------------------------------------------


def test_request_options_sent_only_when_set():
    # Not set -> absent from payload.
    provider, transport = _provider(_response({"role": "assistant", "content": "ok"}))
    provider.complete(None, [Message(role=Role.USER, text="hi")], [])
    assert "reasoning_effort" not in transport.captured
    assert "tool_choice" not in transport.captured
    assert "response_format" not in transport.captured


def test_request_options_forwarded_when_set():
    transport = FakeTransport(_response({"role": "assistant", "content": "ok"}))
    provider = SarvamProvider(
        transport,
        api_key="k",
        reasoning_effort="high",
        tool_choice="auto",
        response_format={"type": "json_object"},
    )
    provider.complete(None, [Message(role=Role.USER, text="hi")], [])
    assert transport.captured["reasoning_effort"] == "high"
    assert transport.captured["tool_choice"] == "auto"
    assert transport.captured["response_format"] == {"type": "json_object"}


# -- offline decode helper --------------------------------------------------


def test_decode_helper_accepts_dict():
    data = _response(
        {"role": "assistant", "content": "hi", "reasoning_content": "r"},
        usage={"prompt_tokens": 7, "completion_tokens": 3},
    )
    turn = SarvamProvider.decode(data)
    assert turn.text == "hi"
    assert turn.reasoning == "r"
    assert turn.usage.input_tokens == 7
    assert turn.usage.output_tokens == 3


def test_decode_helper_accepts_json_string():
    raw = json.dumps(
        _response(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}
                ],
            },
            finish_reason="tool_calls",
        )
    )
    turn = SarvamProvider.decode(raw)
    assert turn.stop_reason is StopReason.TOOL_USE
    assert turn.tool_calls == [ToolCall(id="c1", name="f", input={"x": 1})]
