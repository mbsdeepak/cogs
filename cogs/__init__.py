"""cogs — the core of a coding-agent harness in ~2k readable lines of Python.

A minimal but architecturally real agent runtime: a provider-agnostic agent
loop, a typed tool-call protocol, permission gating, context-window management,
structured tracing, deterministic record/replay, and sub-agents.

Public API::

    from cogs import Agent, AnthropicProvider, ToolRegistry, tool
"""

from __future__ import annotations

from .agent import Agent, AgentError
from .context import ContextManager
from .permissions import Decision, PermissionPolicy, allow_all, gated
from .provider import AnthropicProvider, Provider
from .tools import ToolError, ToolRegistry, tool
from .tools_builtin import default_registry
from .trace import RecordingProvider, ReplayProvider, Tracer
from .types import (
    AssistantTurn,
    Message,
    Role,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentError",
    "AnthropicProvider",
    "Provider",
    "ToolRegistry",
    "tool",
    "ToolError",
    "default_registry",
    "PermissionPolicy",
    "Decision",
    "allow_all",
    "gated",
    "ContextManager",
    "Tracer",
    "ReplayProvider",
    "RecordingProvider",
    "AssistantTurn",
    "Message",
    "Role",
    "StopReason",
    "TextBlock",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "__version__",
]
