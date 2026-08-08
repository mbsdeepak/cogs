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

import json
import os
import urllib.error
import urllib.request
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

# Sarvam AI (OpenAI-compatible Chat Completions). ``sarvam-105b`` is the
# reasoning/agentic model that supports function calling; ``sarvam-m`` is
# deprecated and rejected by the API. See SarvamProvider for the wire mapping.
_SARVAM_ENDPOINT = "https://api.sarvam.ai/v1/chat/completions"
_SARVAM_DEFAULT_MODEL = "sarvam-105b"
# Sarvam's starter tier caps max_tokens at 4096 for sarvam-105b; default to
# that so the common case works without a plan upgrade. Raise via max_tokens=.
_SARVAM_DEFAULT_MAX_TOKENS = 4096

# Sarvam ``finish_reason`` -> normalized StopReason.
_SARVAM_STOP: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}


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


class SarvamProvider:
    """A :class:`Provider` backed by Sarvam AI's Chat Completions API.

    Sarvam is the mandated model partner for the hackathon build. Its API is a
    plain JSON POST with an OpenAI-shaped schema, so — unlike
    :class:`AnthropicProvider` — this needs no third-party SDK at all and talks
    to the endpoint over stdlib ``urllib``, keeping ``cogs`` dependency-free on
    the Sarvam path.

    The OpenAI wire format differs from Anthropic's Messages API in three ways
    that this class translates:

    - the system prompt is the first ``{"role": "system"}`` message, not a
      top-level ``system`` param;
    - a batch of tool results does not ride inside one user turn — each becomes
      its own ``{"role": "tool", "tool_call_id": ...}`` message;
    - tool-call ``arguments`` arrive as a JSON *string* and must be parsed back
      into a dict.

    Auth is the ``api-subscription-key`` header, read from ``SARVAM_API_KEY``.
    Config:

    - ``SARVAM_MODEL``: overrides the default model id (``sarvam-105b``).
    - ``SARVAM_API_KEY``: subscription key from https://dashboard.sarvam.ai.

    ``transport`` is an injection seam mirroring ``AnthropicProvider``'s
    ``client``: a callable ``(payload: dict) -> dict`` that returns a decoded
    Sarvam response. It defaults to a real HTTP POST; tests pass a stub so the
    suite stays offline.
    """

    def __init__(
        self,
        transport: Any | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = _SARVAM_DEFAULT_MAX_TOKENS,
        temperature: float = 0.2,
        endpoint: str | None = None,
        reasoning_effort: str | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self.model = model or os.environ.get("SARVAM_MODEL") or _SARVAM_DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.endpoint = endpoint or _SARVAM_ENDPOINT
        # Optional Chat Completions knobs, sent only when set. ``reasoning_effort``
        # (low/medium/high) tunes sarvam-105b's thinking budget; ``tool_choice``
        # forces/limits tool use; ``response_format`` requests JSON output.
        self.reasoning_effort = reasoning_effort
        self.tool_choice = tool_choice
        self.response_format = response_format
        self._transport = transport or self._http_post

    def complete(
        self,
        system: str | None,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> AssistantTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": self._encode_messages(system, messages),
        }
        if tools:
            payload["tools"] = [self._encode_tool(t) for t in tools]
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.tool_choice is not None:
            payload["tool_choice"] = self.tool_choice
        if self.response_format is not None:
            payload["response_format"] = self.response_format
        return self._decode_response(self._transport(payload))

    def _http_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``payload`` to the Sarvam endpoint and return decoded JSON."""
        if not self.api_key:
            raise RuntimeError(
                "SARVAM_API_KEY is not set; export your Sarvam subscription key, "
                "or pass api_key=... (or a transport= stub for offline use)"
            )
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "api-subscription-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:  # noqa: S310  # trusted host
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"Sarvam API error {exc.code}: {detail}") from exc

    # -- translation: normalized -> OpenAI-style params --------------------

    @staticmethod
    def _encode_messages(
        system: str | None, messages: list[Message]
    ) -> list[dict[str, Any]]:
        """Render the conversation as an OpenAI ``messages`` array."""
        out: list[dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            if m.tool_results:
                # OpenAI splits each result into its own ``tool`` message; the
                # error flag has no wire field, so the content carries it.
                out.extend(
                    {
                        "role": "tool",
                        "tool_call_id": r.call_id,
                        "content": r.content,
                    }
                    for r in m.tool_results
                )
                continue
            if m.tool_calls:
                out.append(
                    {
                        "role": m.role.value,
                        "content": m.text or None,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": json.dumps(c.input),
                                },
                            }
                            for c in m.tool_calls
                        ],
                    }
                )
                continue
            out.append({"role": m.role.value, "content": m.text or ""})
        return out

    @staticmethod
    def _encode_tool(spec: ToolSpec) -> dict[str, Any]:
        """Render one :class:`ToolSpec` as an OpenAI function tool."""
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            },
        }

    # -- translation: response -> normalized -------------------------------

    @classmethod
    def decode(cls, response: dict[str, Any] | str) -> AssistantTurn:
        """Decode a raw Sarvam response (dict or JSON string) offline.

        Public seam for validating the response mapping against a captured
        response — no network, no credits. Accepts either a parsed dict or the
        raw JSON text of a ``chat/completions`` reply.
        """
        if isinstance(response, str):
            response = json.loads(response)
        return cls._decode_response(response)

    @staticmethod
    def _decode_response(data: dict[str, Any]) -> AssistantTurn:
        """Translate a Sarvam response dict into an :class:`AssistantTurn`."""
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or ""
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except (json.JSONDecodeError, AttributeError):
                # Malformed argument JSON from the model: surface an empty dict
                # rather than crash the loop; the tool layer reports bad input.
                args = {}
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), input=args)
            )

        stop_reason = _SARVAM_STOP.get(
            choice.get("finish_reason") or "stop", StopReason.END_TURN
        )
        raw_usage = data.get("usage") or {}
        usage = Usage(
            input_tokens=raw_usage.get("prompt_tokens", 0) or 0,
            output_tokens=raw_usage.get("completion_tokens", 0) or 0,
        )
        # Reasoning models return chain-of-thought in a separate field; keep it
        # apart from the user-facing ``content`` so it never leaks into ``text``.
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        return AssistantTurn(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            reasoning=reasoning,
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
    "SarvamProvider",
    "Role",
    "encode_message",
]
