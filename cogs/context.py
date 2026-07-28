"""Context-window management.

A running conversation grows without bound. This module keeps the estimated
token count under a budget by trimming the *oldest* turns while preserving two
invariants:

1. The most recent turns survive (recency matters most to the model).
2. ``tool_use``/``tool_result`` pairing is never broken — dropping an assistant
   turn that issued tool calls would orphan the following tool-result turn (and
   vice versa), which the Messages API rejects.

The estimator is a deliberately rough char/4 heuristic. Production harnesses
would additionally use server-side compaction (the ``compact-2026-01-12`` beta),
which summarizes rather than drops; this hand-rolled trimmer demonstrates the
mechanics and the pairing constraint that any strategy must respect.
"""

from __future__ import annotations

from .types import Message, Role

# Rough bytes-per-token divisor. Real tokenization is model-specific; this is
# intentionally simple and slightly conservative.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string (char/4, rounded up)."""
    if not text:
        return 0
    return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def estimate_message_tokens(message: Message) -> int:
    """Estimate tokens for one normalized message, including tool payloads."""
    total = estimate_tokens(message.text or "")
    for call in message.tool_calls:
        total += estimate_tokens(call.name)
        total += estimate_tokens(repr(call.input))
    for result in message.tool_results:
        total += estimate_tokens(result.content)
    return total


def _issues_tool_calls(message: Message) -> bool:
    return message.role is Role.ASSISTANT and bool(message.tool_calls)


def _carries_tool_results(message: Message) -> bool:
    return message.role is Role.USER and bool(message.tool_results)


class ContextManager:
    """Trims conversation history to fit a token budget.

    The budget covers the messages only; the system prompt is accounted for
    separately via ``system_tokens`` so it is never trimmed. Trimming removes
    whole turns from the front, but refuses to leave a tool-result turn without
    its preceding tool-call turn — such an orphan is dropped together with, or
    kept together with, its partner.
    """

    def __init__(self, *, max_tokens: int = 100_000, system_tokens: int = 0) -> None:
        self.max_tokens = max_tokens
        self.system_tokens = system_tokens

    def estimate(self, messages: list[Message]) -> int:
        """Total estimated tokens for system prompt + messages."""
        return self.system_tokens + sum(
            estimate_message_tokens(m) for m in messages
        )

    def fit(self, messages: list[Message]) -> list[Message]:
        """Return a suffix of ``messages`` that fits within the budget.

        Drops oldest-first. A tool-result turn is never kept without the
        tool-call turn immediately before it, so the returned list always has
        valid ``tool_use``/``tool_result`` pairing.
        """
        if self.estimate(messages) <= self.max_tokens:
            return list(messages)

        # Walk from the end, accumulating tokens until we'd exceed budget.
        budget = self.max_tokens - self.system_tokens
        kept: list[Message] = []
        running = 0
        for message in reversed(messages):
            cost = estimate_message_tokens(message)
            if running + cost > budget and kept:
                break
            kept.append(message)
            running += cost
        kept.reverse()

        kept = self._repair_pairing(kept, messages)
        return kept

    @staticmethod
    def _repair_pairing(
        kept: list[Message], original: list[Message]
    ) -> list[Message]:
        """Ensure the kept suffix does not start with an orphaned tool-result.

        If the first kept message carries tool results, the assistant turn that
        issued those calls was trimmed away. Drop the orphaned tool-result turn
        so pairing stays valid. Repeat until the head is clean.
        """
        while kept and _carries_tool_results(kept[0]):
            kept = kept[1:]
        return kept


__all__ = [
    "ContextManager",
    "estimate_tokens",
    "estimate_message_tokens",
]
