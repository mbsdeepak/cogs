"""Permission gating for tool execution.

Before a tool runs, the agent consults a :class:`PermissionPolicy`. A tool may
be ``ALLOW`` (run silently), ``DENY`` (never run), or ``ASK`` (defer to a
``confirm`` callback — e.g. a human prompt). A denied or declined tool produces
an errored :class:`~cogs.types.ToolResult` that is fed back to the model, so the
model learns it cannot take that action and can choose another path.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from .types import ToolCall

# A confirmation callback: given the pending call, return True to allow it.
ConfirmFn = Callable[[ToolCall], bool]


class Decision(StrEnum):
    """Per-tool policy outcomes."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


def _deny_all_confirm(_call: ToolCall) -> bool:
    """Default confirm callback: decline everything (safe default)."""
    return False


class PermissionPolicy:
    """A per-tool ``allow``/``ask``/``deny`` policy with a confirm callback.

    ``default`` applies to any tool not named in ``rules``. When a tool resolves
    to :class:`Decision.ASK`, ``confirm`` is invoked; returning ``True`` permits
    the call. The default confirm declines, so ``ASK`` without a supplied
    callback behaves like ``DENY`` — fail closed.
    """

    def __init__(
        self,
        *,
        default: Decision = Decision.ALLOW,
        rules: dict[str, Decision] | None = None,
        confirm: ConfirmFn | None = None,
    ) -> None:
        self.default = default
        self.rules = dict(rules or {})
        self.confirm = confirm or _deny_all_confirm

    def decision_for(self, name: str) -> Decision:
        """The static policy for a tool, before any confirm callback."""
        return self.rules.get(name, self.default)

    def check(self, call: ToolCall) -> tuple[bool, str | None]:
        """Resolve whether ``call`` may run.

        Returns ``(allowed, reason)``. ``reason`` is populated only when the
        call is blocked, and is suitable to surface to the model as error text.
        """
        decision = self.decision_for(call.name)
        if decision is Decision.ALLOW:
            return True, None
        if decision is Decision.DENY:
            return False, f"Tool {call.name!r} is denied by policy."
        # ASK
        if self.confirm(call):
            return True, None
        return False, f"Tool {call.name!r} was declined by the user."


def allow_all() -> PermissionPolicy:
    """A policy that permits every tool. Convenient for tests and demos."""
    return PermissionPolicy(default=Decision.ALLOW)


def gated(
    ask: tuple[str, ...] = ("run_bash", "write_file"),
    *,
    confirm: ConfirmFn | None = None,
) -> PermissionPolicy:
    """An example policy that gates side-effecting tools behind ``ASK``.

    Read-only tools default to ``ALLOW``; the named tools require confirmation.
    """
    return PermissionPolicy(
        default=Decision.ALLOW,
        rules=dict.fromkeys(ask, Decision.ASK),
        confirm=confirm,
    )


__all__ = ["Decision", "PermissionPolicy", "allow_all", "gated", "ConfirmFn"]
