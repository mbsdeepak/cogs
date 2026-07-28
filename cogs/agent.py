"""The agent control loop.

:class:`Agent` owns the pieces that make up an agent's "control plane": a system
prompt, a tool registry, a permission policy, a context manager, a provider, and
a tracer. :meth:`Agent.run` drives the classic tool-use loop to completion:

1. Ask the provider for a completion given the current messages + tools.
2. If the model finished (``end_turn``), return its text.
3. Otherwise, for each requested tool call: gate it through the permission
   policy, execute it (or synthesize an errored result if blocked), and collect
   the results.
4. Append the assistant turn and a *single* user turn carrying **all** tool
   results, then loop.

Sub-agent delegation is exposed as a tool: the parent registers ``spawn_agent``,
which instantiates a child :class:`Agent` with its own registry and context and
returns the child's final answer as the tool result.
"""

from __future__ import annotations

from collections.abc import Callable

from .context import ContextManager, estimate_tokens
from .permissions import PermissionPolicy, allow_all
from .provider import Provider
from .tools import ToolRegistry
from .trace import Tracer
from .types import (
    AssistantTurn,
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolResult,
    Usage,
)

# A hard cap on loop iterations to prevent a pathological model from looping
# forever. Each iteration is one provider call.
_DEFAULT_MAX_STEPS = 24


class AgentError(RuntimeError):
    """Raised when the loop cannot complete (e.g. step budget exhausted)."""


class Agent:
    """A minimal but real coding-agent runtime.

    Parameters mirror the control-plane components. Only ``provider`` is
    required; everything else has a sensible default (empty registry, allow-all
    policy, generous context budget, no-op tracer).
    """

    def __init__(
        self,
        provider: Provider,
        *,
        system: str | None = None,
        registry: ToolRegistry | None = None,
        permissions: PermissionPolicy | None = None,
        context: ContextManager | None = None,
        tracer: Tracer | None = None,
        max_steps: int = _DEFAULT_MAX_STEPS,
    ) -> None:
        self.provider = provider
        self.system = system
        self.registry = registry or ToolRegistry()
        self.permissions = permissions or allow_all()
        self.tracer = tracer or Tracer()
        self.max_steps = max_steps
        self.context = context or ContextManager(
            system_tokens=estimate_tokens(system or "")
        )
        # Aggregate token usage across the whole run.
        self.total_usage = Usage()
        self.messages: list[Message] = []

    # -- public API --------------------------------------------------------

    def run(self, user_message: str) -> str:
        """Drive the loop to completion and return the final assistant text.

        Appends ``user_message`` to the running conversation, then loops until
        the model stops requesting tools. The full history persists on
        ``self.messages`` so :meth:`run` can be called again for a multi-turn
        session.
        """
        self.messages.append(Message(role=Role.USER, text=user_message))
        self.tracer.record_event("run_start", user_message=user_message)

        for step in range(self.max_steps):
            turn = self._complete(step)
            self._accumulate_usage(turn.usage)

            if turn.stop_reason is StopReason.REFUSAL:
                self.tracer.record_event("refusal", step=step)
                return turn.text or "[the model refused to respond]"

            if turn.stop_reason is StopReason.MAX_TOKENS:
                # The turn was truncated. Record it, surface what we have, and
                # stop — retrying is the caller's decision.
                self.tracer.record_event("max_tokens", step=step)
                self._append_assistant(turn)
                return turn.text or "[response truncated at max_tokens]"

            if turn.stop_reason is not StopReason.TOOL_USE:
                # end_turn or stop_sequence: the model is done.
                self._append_assistant(turn)
                self.tracer.record_event("run_end", step=step, text=turn.text)
                return turn.text

            # Tool-use turn: record the assistant message, run the tools, and
            # feed all results back in one user turn.
            self._append_assistant(turn)
            results = [self._execute(call) for call in turn.tool_calls]
            self.messages.append(Message(role=Role.USER, tool_results=results))

        raise AgentError(
            f"agent did not finish within {self.max_steps} steps; "
            "the model kept requesting tools."
        )

    def register_subagent_tool(
        self,
        *,
        name: str = "spawn_agent",
        system: str | None = None,
        registry: ToolRegistry | None = None,
        permissions: PermissionPolicy | None = None,
    ) -> None:
        """Register a tool that delegates a task to a child :class:`Agent`.

        The child runs its own loop with its own registry/context against the
        same provider, and returns its final answer as this tool's result. This
        demonstrates sub-agent delegation: the parent's context stays lean while
        the child does focused work.
        """
        child_system = system or "You are a focused sub-agent. Complete the task and report back concisely."
        child_registry = registry or ToolRegistry()
        child_permissions = permissions or self.permissions

        def spawn_agent(task: str) -> str:
            """Delegate a self-contained task to a fresh sub-agent.

            Args:
                task: A complete, self-contained description of the work.
            """
            child = Agent(
                self.provider,
                system=child_system,
                registry=child_registry,
                permissions=child_permissions,
                tracer=self.tracer,
            )
            self.tracer.record_event("subagent_start", task=task)
            answer = child.run(task)
            self._accumulate_usage(child.total_usage)
            self.tracer.record_event("subagent_end", answer=answer)
            return answer

        self.registry.register(spawn_agent, name=name)

    # -- internals ---------------------------------------------------------

    def _complete(self, step: int) -> AssistantTurn:
        """Fit the context to budget and request one completion."""
        self.messages = self.context.fit(self.messages)
        turn = self.provider.complete(
            self.system, self.messages, self.registry.specs()
        )
        self.tracer.record_provider_call(step, turn)
        return turn

    def _append_assistant(self, turn: AssistantTurn) -> None:
        self.messages.append(
            Message(
                role=Role.ASSISTANT,
                text=turn.text or None,
                tool_calls=list(turn.tool_calls),
            )
        )

    def _execute(self, call: ToolCall) -> ToolResult:
        """Gate and execute a single tool call.

        A blocked call (denied or declined) becomes an errored result without
        ever invoking the tool body; the model sees the reason and can adapt.
        """
        allowed, reason = self.permissions.check(call)
        if not allowed:
            result = ToolResult(
                call_id=call.id, content=f"Error: {reason}", is_error=True
            )
            self.tracer.record_event(
                "tool_blocked", tool=call.name, reason=reason
            )
            return result
        result = self.registry.dispatch(call)
        self.tracer.record_tool_execution(call, result)
        return result

    def _accumulate_usage(self, usage: Usage) -> None:
        self.total_usage = Usage(
            input_tokens=self.total_usage.input_tokens + usage.input_tokens,
            output_tokens=self.total_usage.output_tokens + usage.output_tokens,
            cache_read_tokens=self.total_usage.cache_read_tokens
            + usage.cache_read_tokens,
            cache_creation_tokens=self.total_usage.cache_creation_tokens
            + usage.cache_creation_tokens,
        )


Confirm = Callable[[ToolCall], bool]

__all__ = ["Agent", "AgentError"]
