"""Run a cogs agent against the env-configured provider (real Bedrock).

Usage:

    AWS_REGION=us-east-1 python examples/coding_agent.py "List the files here and summarize the README."

Environment:
    COGS_PROVIDER   bedrock (default) | anthropic
    COGS_MODEL      overrides the default model id
    AWS_REGION      Bedrock region (default us-east-1)

This talks to the real API and records a cassette to ``coding_agent.jsonl`` that
you can replay offline (see examples/replay_demo.py). Requires the ``anthropic``
SDK and valid credentials for the chosen provider.
"""

from __future__ import annotations

import sys

from cogs import Agent, AnthropicProvider, Decision, PermissionPolicy, Tracer
from cogs.tools_builtin import default_registry
from cogs.trace import RecordingProvider

SYSTEM = (
    "You are cogs, a concise coding agent. Use the tools to inspect files. "
    "Read before you conclude, and answer plainly."
)


def main() -> int:
    task = " ".join(sys.argv[1:]) or "List the files in the current directory."

    # Auto-approve gated tools in this non-interactive example.
    policy = PermissionPolicy(
        default=Decision.ALLOW,
        rules={"write_file": Decision.ASK, "run_bash": Decision.ASK},
        confirm=lambda call: True,
    )

    with Tracer("coding_agent.jsonl") as tracer:
        provider = RecordingProvider(AnthropicProvider(), tracer)
        agent = Agent(
            provider,
            system=SYSTEM,
            registry=default_registry(),
            permissions=policy,
            tracer=tracer,
        )
        answer = agent.run(task)

    print(answer)
    u = agent.total_usage
    print(f"\n[usage] in={u.input_tokens} out={u.output_tokens} total={u.total_tokens}")
    print("[trace] recorded to coding_agent.jsonl (replay it offline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
