"""Command-line entry point: ``python -m cogs`` or the ``cogs`` script.

Two modes:

- Interactive REPL against the env-configured provider (real Bedrock by
  default). Prints per-turn token usage.
- ``--replay <cassette>``: drive the same agent loop against a recorded
  cassette with :class:`~cogs.trace.ReplayProvider` — zero credentials, zero
  network.
"""

from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .permissions import Decision, PermissionPolicy
from .provider import AnthropicProvider, Provider
from .tools_builtin import default_registry
from .trace import ReplayProvider, Tracer
from .types import ToolCall

_SYSTEM_PROMPT = (
    "You are cogs, a concise coding agent. Use the available tools to inspect "
    "and modify files. Prefer read-only tools first. When you have the answer, "
    "state it plainly."
)


def _cli_confirm(call: ToolCall) -> bool:
    """Prompt the operator to approve a gated tool call at the terminal."""
    print(f"\n[permission] allow {call.name}({call.input})? [y/N] ", end="")
    try:
        answer = input().strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _build_agent(provider: Provider, tracer: Tracer) -> Agent:
    policy = PermissionPolicy(
        default=Decision.ALLOW,
        rules={"write_file": Decision.ASK, "run_bash": Decision.ASK},
        confirm=_cli_confirm,
    )
    return Agent(
        provider,
        system=_SYSTEM_PROMPT,
        registry=default_registry(),
        permissions=policy,
        tracer=tracer,
    )


def _print_usage(agent: Agent) -> None:
    u = agent.total_usage
    print(
        f"  [usage] in={u.input_tokens} out={u.output_tokens} "
        f"cache_read={u.cache_read_tokens} total={u.total_tokens}",
        file=sys.stderr,
    )


def _run_replay(cassette: str) -> int:
    provider = ReplayProvider.from_cassette(cassette)
    with Tracer() as tracer:
        agent = _build_agent(provider, tracer)
        # A replay ignores the user text, but the loop still needs a kickoff.
        answer = agent.run("(replaying recorded session)")
    print(answer)
    _print_usage(agent)
    return 0


def _run_interactive() -> int:
    provider = AnthropicProvider()
    print(
        f"cogs REPL — provider={provider.provider} model={provider.model}. "
        "Type a message, or Ctrl-D to exit.",
        file=sys.stderr,
    )
    with Tracer("cogs-trace.jsonl") as tracer:
        agent = _build_agent(provider, tracer)
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            answer = agent.run(line)
            print(answer)
            _print_usage(agent)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cogs", description="A minimal agent runtime.")
    parser.add_argument(
        "--replay",
        metavar="CASSETTE",
        help="replay a recorded JSONL cassette offline (no credentials needed)",
    )
    args = parser.parse_args(argv)

    if args.replay:
        return _run_replay(args.replay)
    return _run_interactive()


if __name__ == "__main__":
    raise SystemExit(main())
