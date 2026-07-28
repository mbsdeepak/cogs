"""Provider translation, offline. No SDK client, no network.

We inject a fake client that mimics the ``.messages.create`` surface and
returns objects shaped like the anthropic SDK response, then assert on both the
request params we send and the normalized turn we decode.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cogs.provider import AnthropicProvider
from cogs.types import Message, Role, StopReason, ToolCall, ToolResult, ToolSpec


class FakeClient:
    """Captures create() params and returns a canned response."""

    def __init__(self, response) -> None:
        self._response = response
        self.captured = None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **params):
        self.captured = params
        return self._response


def _response(content, stop_reason="end_turn", usage=None):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=usage
        or SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


def test_decode_text_response():
    resp = _response([SimpleNamespace(type="text", text="hello")])
    provider = AnthropicProvider(FakeClient(resp), provider="anthropic")
    turn = provider.complete("sys", [Message(role=Role.USER, text="hi")], [])
    assert turn.text == "hello"
    assert turn.stop_reason is StopReason.END_TURN
    assert turn.usage.input_tokens == 100


def test_decode_tool_use_response():
    resp = _response(
        [
            SimpleNamespace(type="text", text="let me check"),
            SimpleNamespace(
                type="tool_use", id="tu1", name="get_x", input={"q": "v"}
            ),
        ],
        stop_reason="tool_use",
    )
    provider = AnthropicProvider(FakeClient(resp), provider="anthropic")
    turn = provider.complete(None, [Message(role=Role.USER, text="hi")], [])
    assert turn.stop_reason is StopReason.TOOL_USE
    assert turn.text == "let me check"
    assert turn.tool_calls == [ToolCall(id="tu1", name="get_x", input={"q": "v"})]


def test_encode_plain_user_message():
    client = FakeClient(_response([SimpleNamespace(type="text", text="ok")]))
    provider = AnthropicProvider(client, provider="anthropic")
    provider.complete("system prompt", [Message(role=Role.USER, text="hi")], [])
    params = client.captured
    assert params["system"] == "system prompt"
    assert params["messages"] == [{"role": "user", "content": "hi"}]
    # Opus 4.8: no sampling params, no thinking by default.
    assert "temperature" not in params
    assert "top_p" not in params
    assert "thinking" not in params


def test_encode_tool_result_message():
    client = FakeClient(_response([SimpleNamespace(type="text", text="ok")]))
    provider = AnthropicProvider(client, provider="anthropic")
    msg = Message(
        role=Role.USER,
        tool_results=[ToolResult(call_id="t1", content="42", is_error=False)],
    )
    provider.complete(None, [msg], [])
    content = client.captured["messages"][0]["content"]
    assert content == [
        {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": "42",
            "is_error": False,
        }
    ]


def test_encode_assistant_tool_call_message():
    client = FakeClient(_response([SimpleNamespace(type="text", text="ok")]))
    provider = AnthropicProvider(client, provider="anthropic")
    msg = Message(
        role=Role.ASSISTANT,
        text="calling",
        tool_calls=[ToolCall(id="t1", name="add", input={"a": 1})],
    )
    provider.complete(None, [msg], [])
    content = client.captured["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "calling"}
    assert content[1] == {
        "type": "tool_use",
        "id": "t1",
        "name": "add",
        "input": {"a": 1},
    }


def test_tools_serialized_into_request():
    client = FakeClient(_response([SimpleNamespace(type="text", text="ok")]))
    provider = AnthropicProvider(client, provider="anthropic")
    spec = ToolSpec(
        name="add",
        description="Add.",
        input_schema={"type": "object", "properties": {}},
    )
    provider.complete(None, [Message(role=Role.USER, text="hi")], [spec])
    assert client.captured["tools"] == [
        {"name": "add", "description": "Add.", "input_schema": {"type": "object", "properties": {}}}
    ]


def test_thinking_opt_in_adds_adaptive():
    client = FakeClient(_response([SimpleNamespace(type="text", text="ok")]))
    provider = AnthropicProvider(client, provider="anthropic", thinking=True)
    provider.complete(None, [Message(role=Role.USER, text="hi")], [])
    assert client.captured["thinking"] == {"type": "adaptive"}


def test_default_model_bedrock_prefix():
    client = FakeClient(_response([SimpleNamespace(type="text", text="ok")]))
    provider = AnthropicProvider(client, provider="bedrock")
    assert provider.model == "anthropic.claude-opus-4-8"


def test_default_model_anthropic_no_prefix():
    client = FakeClient(_response([SimpleNamespace(type="text", text="ok")]))
    provider = AnthropicProvider(client, provider="anthropic")
    assert provider.model == "claude-opus-4-8"


def test_explicit_model_override():
    client = FakeClient(_response([SimpleNamespace(type="text", text="ok")]))
    provider = AnthropicProvider(
        client, provider="bedrock", model="anthropic.claude-custom"
    )
    assert provider.model == "anthropic.claude-custom"


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="unknown COGS_PROVIDER"):
        AnthropicProvider(object(), provider="openai")


def test_max_tokens_default_and_override():
    client = FakeClient(_response([SimpleNamespace(type="text", text="ok")]))
    provider = AnthropicProvider(client, provider="anthropic")
    provider.complete(None, [Message(role=Role.USER, text="hi")], [])
    assert client.captured["max_tokens"] == 8192
