"""Tool schema generation and dispatch."""

from __future__ import annotations

from cogs.tools import ToolError, ToolRegistry, build_spec, tool
from cogs.types import ToolCall


def test_schema_generation_basic_types():
    def sample(name: str, count: int, ratio: float, flag: bool) -> str:
        """Do a thing.

        Args:
            name: the name
            count: how many
        """
        return "ok"

    spec = build_spec(sample)
    props = spec.input_schema["properties"]
    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["ratio"]["type"] == "number"
    assert props["flag"]["type"] == "boolean"
    assert spec.description == "Do a thing."


def test_schema_param_descriptions_from_docstring():
    def sample(name: str, count: int) -> str:
        """Summary line.

        Args:
            name: the display name
            count: number of items
        """
        return ""

    spec = build_spec(sample)
    props = spec.input_schema["properties"]
    assert props["name"]["description"] == "the display name"
    assert props["count"]["description"] == "number of items"


def test_schema_required_vs_optional():
    def sample(required_arg: str, optional_arg: int = 3) -> str:
        """S."""
        return ""

    spec = build_spec(sample)
    assert spec.input_schema["required"] == ["required_arg"]
    assert "optional_arg" in spec.input_schema["properties"]
    assert "optional_arg" not in spec.input_schema["required"]


def test_schema_no_required_key_when_all_optional():
    def sample(a: str = "x") -> str:
        """S."""
        return ""

    spec = build_spec(sample)
    assert "required" not in spec.input_schema


def test_schema_list_type():
    def sample(items: list[int]) -> str:
        """S."""
        return ""

    spec = build_spec(sample)
    prop = spec.input_schema["properties"]["items"]
    assert prop["type"] == "array"
    assert prop["items"]["type"] == "integer"


def test_schema_bare_list_type():
    def sample(items: list) -> str:
        """S."""
        return ""

    spec = build_spec(sample)
    prop = spec.input_schema["properties"]["items"]
    assert prop["type"] == "array"
    assert "items" not in prop


def test_tool_decorator_attaches_spec():
    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"hi {name}"

    assert greet.tool_spec.name == "greet"
    # Still directly callable.
    assert greet("world") == "hi world"


def test_missing_annotation_defaults_to_string():
    def sample(x) -> str:  # no annotation
        """S."""
        return ""

    spec = build_spec(sample)
    assert spec.input_schema["properties"]["x"]["type"] == "string"


def test_registry_dispatch_success():
    reg = ToolRegistry()

    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    reg.register(add)
    result = reg.dispatch(ToolCall(id="1", name="add", input={"a": 2, "b": 3}))
    assert result.content == "5"
    assert result.is_error is False
    assert result.call_id == "1"


def test_registry_dispatch_unknown_tool():
    reg = ToolRegistry()
    result = reg.dispatch(ToolCall(id="1", name="nope", input={}))
    assert result.is_error is True
    assert "unknown tool" in result.content


def test_registry_dispatch_bad_arguments():
    reg = ToolRegistry()

    @tool
    def needs_two(a: int, b: int) -> int:
        """Add."""
        return a + b

    reg.register(needs_two)
    result = reg.dispatch(ToolCall(id="1", name="needs_two", input={"a": 1}))
    assert result.is_error is True
    assert "invalid arguments" in result.content


def test_registry_dispatch_captures_exception():
    reg = ToolRegistry()

    @tool
    def boom() -> str:
        """Boom."""
        raise RuntimeError("kaboom")

    reg.register(boom)
    result = reg.dispatch(ToolCall(id="1", name="boom", input={}))
    assert result.is_error is True
    assert "kaboom" in result.content


def test_registry_dispatch_tool_error_message():
    reg = ToolRegistry()

    @tool
    def strict(x: int) -> str:
        """Strict."""
        raise ToolError("nope, bad x")

    reg.register(strict)
    result = reg.dispatch(ToolCall(id="1", name="strict", input={"x": 1}))
    assert result.is_error is True
    assert "nope, bad x" in result.content


def test_registry_rejects_duplicate():
    reg = ToolRegistry()

    @tool
    def dup() -> str:
        """D."""
        return ""

    reg.register(dup)
    try:
        reg.register(dup)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_registry_specs_order_preserved():
    reg = ToolRegistry()

    @tool
    def a_tool() -> str:
        """A."""
        return ""

    @tool
    def b_tool() -> str:
        """B."""
        return ""

    reg.register(a_tool)
    reg.register(b_tool)
    assert [s.name for s in reg.specs()] == ["a_tool", "b_tool"]
