"""Provider abstraction: the only place the ``anthropic`` SDK is imported.

The agent loop is provider-agnostic. It calls :meth:`Provider.complete` with
normalized messages and tool specs and receives a normalized
:class:`~cogs.types.AssistantTurn`. Everything SDK-specific — client
construction, wire-format translation, model-id prefixing — is isolated here.

The whole architectural story of ``cogs`` is that swapping providers is a
one-line change in client construction. The official ``anthropic`` SDK ships an
``AnthropicBedrock`` client whose ``.messages.create(...)`` surface is identical
to the direct ``Anthropic`` client, so the loop below never has to know which
one it is talking to.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from .types import (
    AssistantTurn,
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolSpec,
    Usage,
)

# Defaults per provider. On Bedrock, model ids take an ``anthropic.`` prefix.
_DEFAULT_MODEL = {
    "bedrock": "anthropic.claude-opus-4-8",
    "anthropic": "claude-opus-4-8",
}
_DEFAULT_MAX_TOKENS = 8192


@runtime_checkable
class Provider(Protocol):
    """A source of assistant turns.

    Implementations translate normalized messages + tool specs into a single
    model completion. :class:`AnthropicProvider` talks to the real API;
    :class:`~cogs.trace.ReplayProvider` replays a recorded cassette.
    """

    def complete(
        self,
        system: str | None,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> AssistantTurn:
        """Return one assistant turn for the given conversation state."""
        ...


class AnthropicProvider:
    """A :class:`Provider` backed by the official ``anthropic`` SDK.

    Accepts an already-constructed client (useful for tests and custom auth) or
    builds one from environment variables:

    - ``COGS_PROVIDER``: ``bedrock`` (default) or ``anthropic``.
    - ``COGS_MODEL``: overrides the per-provider default model id.
    - ``AWS_REGION``: region for the Bedrock client (default ``us-east-1``).

    On Opus 4.8, sampling params (``temperature``/``top_p``/``top_k``) are
    rejected by the API, so we never send them. Adaptive thinking is opt-in via
    ``thinking=True`` and left OFF by default for maximum Bedrock compatibility.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str | None = None,
        provider: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        thinking: bool = False,
    ) -> None:
        provider = provider or os.environ.get("COGS_PROVIDER", "bedrock")
        if provider not in _DEFAULT_MODEL:
            raise ValueError(
                f"unknown COGS_PROVIDER {provider!r}; "
                f"expected one of {sorted(_DEFAULT_MODEL)}"
            )
        self.provider = provider
        self.model = model or os.environ.get("COGS_MODEL") or _DEFAULT_MODEL[provider]
        self.max_tokens = max_tokens
        self.thinking = thinking
        self._client = client if client is not None else self._build_client(provider)

    @staticmethod
    def _build_client(provider: str) -> Any:
        """Construct the SDK client for ``provider`` from the environment.

        Importing ``anthropic`` lazily keeps it out of the import path for
        offline replay-only usage (tests, the replay demo).
        """
        if provider == "bedrock":
            from anthropic import AnthropicBedrock

            return AnthropicBedrock(
                aws_region=os.environ.get("AWS_REGION", "us-east-1")
            )
        from anthropic import Anthropic

        return Anthropic()

    def complete(
        self,
        system: str | None,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> AssistantTurn:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [self._encode_message(m) for m in messages],
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [t.to_api() for t in tools]
        if self.thinking:
            # Opt-in; see class docstring. Bedrock support varies by region.
            params["thinking"] = {"type": "adaptive"}

        response = self._client.messages.create(**params)
        return self._decode_response(response)

    # -- translation: normalized -> SDK params -----------------------------

    @staticmethod
    def _encode_message(message: Message) -> dict[str, Any]:
        """Render one normalized :class:`Message` as a Messages API dict."""
        if message.tool_results:
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": r.call_id,
                    "content": r.content,
                    "is_error": r.is_error,
                }
                for r in message.tool_results
            ]
            return {"role": message.role.value, "content": content}

        if message.tool_calls:
            content: list[dict[str, Any]] = []
            if message.text:
                content.append({"type": "text", "text": message.text})
            content.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input}
                for c in message.tool_calls
            )
            return {"role": message.role.value, "content": content}

        return {"role": message.role.value, "content": message.text or ""}

    # -- translation: SDK response -> normalized ---------------------------

    @staticmethod
    def _decode_response(response: Any) -> AssistantTurn:
        """Translate an SDK response object into an :class:`AssistantTurn`."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )

        raw_usage = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=getattr(raw_usage, "input_tokens", 0) or 0,
            output_tokens=getattr(raw_usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(raw_usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(
                raw_usage, "cache_creation_input_tokens", 0
            )
            or 0,
        )
        return AssistantTurn(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=StopReason(response.stop_reason),
            usage=usage,
        )


def encode_message(message: Message) -> dict[str, Any]:
    """Public helper mirroring the provider's message encoding.

    Exposed so callers (and the tracer) can serialize a conversation without
    holding a provider instance. Kept in sync with
    :meth:`AnthropicProvider._encode_message`.
    """
    return AnthropicProvider._encode_message(message)


__all__ = [
    "Provider",
    "AnthropicProvider",
    "Role",
    "encode_message",
]
