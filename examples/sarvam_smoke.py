"""Live smoke test: does Sarvam actually drive the cogs tool-use loop?

This is the *load-bearing* de-risking check for using Sarvam as the model
partner: the whole platform assumes native function calling works. It runs two
probes against the real API:

  A. Raw provider probe — send one tool spec and a prompt that should force a
     call, then report whether ``sarvam-105b`` returned a structured
     ``tool_call`` with parseable arguments.
  B. Full agent loop — run a real cogs Agent with the built-in tools end to end
     over Sarvam, proving the whole runtime is provider-swappable.

Usage:

    export SARVAM_API_KEY=sk_...            # from https://dashboard.sarvam.ai
    python examples/sarvam_smoke.py         # runs probe A, then probe B
    python examples/sarvam_smoke.py --raw   # probe A only

Environment:
    SARVAM_API_KEY   Sarvam subscription key (required)
    SARVAM_MODEL     overrides the default model id (sarvam-105b)

Exit code is non-zero if probe A does not produce a tool call, so this doubles
as a CI/pre-hackathon gate: if it fails, cogs needs a JSON/ReAct fallback.
"""

from __future__ import annotations

import sys

from cogs import Agent, Decision, PermissionPolicy, SarvamProvider
from cogs.tools_builtin import default_registry
from cogs.types import Message, Role, StopReason, ToolSpec

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
)


def probe_raw(provider: SarvamProvider) -> bool:
    """Return True iff Sarvam emits a well-formed tool call."""
    print(f"[A] raw tool-call probe against model={provider.model!r} ...")
    turn = provider.complete(
        system="You are a weather assistant. Use the get_weather tool.",
        messages=[Message(role=Role.USER, text="What's the weather in Bengaluru?")],
        tools=[WEATHER_TOOL],
    )
    print(f"    stop_reason = {turn.stop_reason.value}")
    print(f"    text        = {turn.text!r}")
    if not turn.tool_calls:
        print("    ✗ no tool_calls returned — Sarvam did not call the tool")
        return False
    for call in turn.tool_calls:
        print(f"    ✓ tool_call: name={call.name!r} input={call.input!r}")
    ok = turn.stop_reason is StopReason.TOOL_USE and turn.tool_calls[0].name == "get_weather"
    print(f"    tokens in={turn.usage.input_tokens} out={turn.usage.output_tokens}")
    return ok


def probe_agent_loop(provider: SarvamProvider) -> None:
    """Drive a full cogs Agent over Sarvam with the built-in tools."""
    print("\n[B] full agent loop over Sarvam ...")
    policy = PermissionPolicy(
        default=Decision.ALLOW,
        rules={"write_file": Decision.ASK, "run_bash": Decision.ASK},
        confirm=lambda call: True,
    )
    agent = Agent(
        provider,
        system=(
            "You are cogs, a concise coding agent. Use the tools to inspect "
            "files. Read before you conclude, and answer plainly."
        ),
        registry=default_registry(),
        permissions=policy,
    )
    answer = agent.run("List the files in the current directory and count them.")
    print(f"    answer: {answer}")
    u = agent.total_usage
    print(f"    [usage] in={u.input_tokens} out={u.output_tokens} total={u.total_tokens}")


def main() -> int:
    provider = SarvamProvider()  # reads SARVAM_API_KEY / SARVAM_MODEL from env
    raw_only = "--raw" in sys.argv[1:]

    ok = probe_raw(provider)
    if not ok:
        print("\nRESULT: ✗ tool calling did NOT work — add a JSON/ReAct fallback.")
        return 1

    if not raw_only:
        probe_agent_loop(provider)

    print("\nRESULT: ✓ Sarvam tool calling works with cogs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
