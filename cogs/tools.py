"""Tool definition + dispatch.

The :func:`tool` decorator turns a plain Python function into a registered tool
by introspecting its type hints and docstring to synthesize a JSON-Schema
:class:`~cogs.types.ToolSpec`. A :class:`ToolRegistry` holds the specs and the
callables and dispatches calls by name, capturing any exception into an errored
:class:`~cogs.types.ToolResult` so a buggy tool never crashes the agent loop.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_args, get_origin, get_type_hints

from .types import ToolCall, ToolResult, ToolSpec

# Python type -> JSON Schema type. ``list`` maps to an untyped array; a
# parameterized ``list[int]`` additionally sets ``items``.
_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
}


class ToolError(Exception):
    """Raised by tool bodies to signal a recoverable, model-visible failure."""


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into (summary, {param: description}).

    The first non-empty line is the summary. Lines under an ``Args:`` section of
    the form ``name: description`` populate per-parameter descriptions.
    """
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    summary = lines[0].strip() if lines else ""

    params: dict[str, str] = {}
    in_args = False
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.rstrip(":").lower() in ("args", "arguments", "parameters"):
            in_args = True
            continue
        if in_args:
            if not stripped or stripped.endswith(":") and " " not in stripped:
                # A new section header ends the Args block.
                if stripped and stripped.endswith(":"):
                    break
                continue
            if ":" in stripped:
                name, _, desc = stripped.partition(":")
                params[name.strip()] = desc.strip()
    return summary, params


def _json_schema_for(annotation: Any) -> dict[str, Any]:
    """Map a type annotation to a JSON-Schema fragment."""
    origin = get_origin(annotation)
    if origin in (list, list):
        schema: dict[str, Any] = {"type": "array"}
        args = get_args(annotation)
        if args:
            schema["items"] = _json_schema_for(args[0])
        return schema
    json_type = _JSON_TYPES.get(annotation)
    if json_type is None:
        # Fall back to string for unsupported/unknown annotations.
        return {"type": "string"}
    return {"type": json_type}


def build_spec(func: Callable[..., Any], name: str | None = None) -> ToolSpec:
    """Introspect ``func`` and build its :class:`~cogs.types.ToolSpec`.

    Required parameters are those without a default. Optional parameters (those
    with a default) are omitted from ``required`` but still described.
    """
    summary, param_docs = _parse_docstring(func.__doc__)
    sig = inspect.signature(func)
    # Resolve string annotations (from ``from __future__ import annotations``)
    # into real types. Fall back to the raw signature if resolution fails.
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001 - best-effort; tolerate unresolvable hints
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(pname, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = str
        prop = _json_schema_for(annotation)
        if pname in param_docs:
            prop["description"] = param_docs[pname]
        properties[pname] = prop
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return ToolSpec(
        name=name or func.__name__,
        description=summary or f"Tool {name or func.__name__}",
        input_schema=schema,
    )


def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator: attach a generated :class:`~cogs.types.ToolSpec` to ``func``.

    The spec is available as ``func.tool_spec``. Decorated functions are plain
    callables — you can still call them directly in tests.
    """
    func.tool_spec = build_spec(func)  # type: ignore[attr-defined]
    return func


class ToolRegistry:
    """A named collection of tools with validated dispatch."""

    def __init__(self) -> None:
        self._funcs: dict[str, Callable[..., Any]] = {}
        self._specs: dict[str, ToolSpec] = {}

    def register(
        self, func: Callable[..., Any], *, name: str | None = None
    ) -> Callable[..., Any]:
        """Add a tool. Uses ``func.tool_spec`` if present, else builds one."""
        spec: ToolSpec = getattr(func, "tool_spec", None) or build_spec(func, name)
        if name and name != spec.name:
            spec = ToolSpec(name, spec.description, spec.input_schema)
        if spec.name in self._funcs:
            raise ValueError(f"tool {spec.name!r} already registered")
        self._funcs[spec.name] = func
        self._specs[spec.name] = spec
        return func

    def specs(self) -> list[ToolSpec]:
        """All registered specs, in registration order."""
        return list(self._specs.values())

    def names(self) -> list[str]:
        return list(self._funcs)

    def __contains__(self, name: str) -> bool:
        return name in self._funcs

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Execute ``call`` and return a :class:`~cogs.types.ToolResult`.

        All failure modes — unknown tool, bad arguments, raised exception — are
        captured as an errored result and returned, never raised. The model
        sees the error text and can adapt.
        """
        func = self._funcs.get(call.name)
        if func is None:
            return ToolResult(
                call_id=call.id,
                content=f"Error: unknown tool {call.name!r}.",
                is_error=True,
            )
        try:
            result = func(**call.input)
        except TypeError as exc:
            return ToolResult(
                call_id=call.id,
                content=f"Error: invalid arguments for {call.name!r}: {exc}",
                is_error=True,
            )
        except ToolError as exc:
            return ToolResult(call_id=call.id, content=f"Error: {exc}", is_error=True)
        except Exception as exc:  # noqa: BLE001 - tools must never crash the loop
            return ToolResult(
                call_id=call.id,
                content=f"Error: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        return ToolResult(call_id=call.id, content=str(result), is_error=False)


__all__ = ["tool", "build_spec", "ToolRegistry", "ToolError"]
