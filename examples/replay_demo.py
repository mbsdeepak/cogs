"""Run the SAME agent loop against a checked-in cassette — no credentials.

    python examples/replay_demo.py

This drives the full agent loop (message assembly, tool execution, permission
gating) using :class:`~cogs.trace.ReplayProvider`, which returns recorded model
turns instead of calling the API. It proves the loop works end-to-end with zero
network access — the backbone of the offline test suite.
"""

from __future__ import annotations

from pathlib import Path

from cogs import Agent, ReplayProvider, Tracer, allow_all
from cogs.tools_builtin import default_registry

CASSETTE = Path(__file__).parent / "cassettes" / "coding_session.jsonl"

SYSTEM = "You are cogs, a concise coding agent."


def main() -> int:
    provider = ReplayProvider.from_cassette(CASSETTE)
    with Tracer() as tracer:
        agent = Agent(
            provider,
            system=SYSTEM,
            registry=default_registry(),
            permissions=allow_all(),
            tracer=tracer,
        )
        answer = agent.run("How many Python files are in the cogs package, and what does types.py define?")

    print("=== final answer ===")
    print(answer)
    print("\n=== tool executions (from trace) ===")
    for event in tracer.events:
        if event["event"] == "tool_execution":
            print(f"  {event['name']}({event['input']}) -> {event['result']['content'][:60]!r}")
    u = agent.total_usage
    print(f"\n[usage] in={u.input_tokens} out={u.output_tokens} total={u.total_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
