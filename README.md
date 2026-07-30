# cogs

**The core of a coding-agent harness — like Claude Code — in ~2k readable lines of Python.**

`cogs` is a minimal but architecturally real agent runtime. It is not a toy loop
and not a framework: it is a tight, legible implementation of the *control
plane* every coding agent needs — the agent loop, a typed tool-call protocol, a
provider abstraction, context-window management, permission gating, structured
tracing, deterministic record/replay, and sub-agent delegation.

It exists to demonstrate that you understand how a harness actually works, down
to the message-assembly rules and the `tool_use`/`tool_result` pairing
invariant, without hiding behind a dependency.

---

## Why this exists / what it demonstrates

The LLM-agent ecosystem is full of frameworks that abstract the loop away. This
repo does the opposite: it makes the loop the point. Reading it should teach you

- **the agent loop shape** — call the model; if it asked for tools, run them,
  feed all results back in a single turn, repeat until it stops;
- **why providers are swappable** — the loop speaks normalized dataclasses, and
  the only file that imports the vendor SDK is `provider.py`. Switching from AWS
  Bedrock to a direct API key is a one-line change in client construction;
- **the boring-but-load-bearing invariants** — every tool call gets exactly one
  result; all results for one assistant turn go in one user message; context
  trimming must never orphan a `tool_result`;
- **how to test an agent without a network** — record real model turns to a
  JSONL "cassette", then replay them deterministically. The entire test suite
  and the demo run offline.

---

## Architecture

```
                                   ┌───────────────────────────────┐
                                   │            Agent.run          │
   user message ───────────────▶  │        (the agent loop)       │
                                   └───────────────┬───────────────┘
                                                   │ normalized Messages + ToolSpecs
                        ┌──────────────────────────┼──────────────────────────┐
                        ▼                          ▼                          ▼
                 ┌─────────────┐          ┌────────────────┐         ┌────────────────┐
                 │  Provider   │          │ PermissionPolicy│         │ ContextManager │
                 │ (abstract)  │          │ allow/ask/deny  │         │  token budget  │
                 └──────┬──────┘          └────────────────┘         └────────────────┘
              ┌─────────┴──────────┐
              ▼                    ▼
      ┌───────────────┐    ┌────────────────┐
      │ AnthropicProv │    │ ReplayProvider │   ← replays a cassette, zero network
      │ Bedrock | API │    │  (offline)     │
      └───────┬───────┘    └────────────────┘
              │ .messages.create(...)              every call + tool run is
              ▼                                     appended to a JSONL cassette
   ┌──────────────────────┐                         by the Tracer  ─────────────┐
   │  anthropic SDK        │                                                     ▼
   │  AnthropicBedrock  ── identical .messages surface ──  Anthropic()   ┌──────────────┐
   └──────────────────────┘                                              │  cassette    │
                                                                         │  .jsonl      │
        loop:  complete → (tool_use?) → gate → dispatch → tool_results ──┘  record/replay
               → append assistant turn + one user turn of results → repeat until end_turn
```

The provider boundary is the whole trick: `AnthropicBedrock` and `Anthropic`
expose the *identical* `.messages.create(...)` surface, so the loop is written
once and runs against either.

---

## Module map

| Module               | Responsibility |
|----------------------|----------------|
| `cogs/types.py`      | Normalized dataclasses the loop speaks: `Message`, `ToolCall`, `ToolResult`, `AssistantTurn`, `Usage`, `ToolSpec`. Providers translate to/from these. |
| `cogs/provider.py`   | `Provider` protocol + `AnthropicProvider`. The **only** file that imports `anthropic`. Builds a Bedrock or direct client from env and translates messages ⇄ SDK. |
| `cogs/tools.py`      | The `@tool` decorator (JSON-Schema from type hints + docstring) and `ToolRegistry` (validated dispatch, exceptions → errored results). |
| `cogs/permissions.py`| `PermissionPolicy` with per-tool `allow`/`ask`/`deny` and a pluggable `confirm` callback. |
| `cogs/context.py`    | Token estimation + a trimming strategy that preserves recency and never breaks `tool_use`/`tool_result` pairing. |
| `cogs/trace.py`      | `Tracer` (JSONL cassettes), `RecordingProvider` (wrap+record), `ReplayProvider` (deterministic offline replay). |
| `cogs/agent.py`      | `Agent` — owns the control-plane pieces and drives `run()`. Includes sub-agent delegation as a tool. |
| `cogs/tools_builtin.py` | Example tools: `read_file`, `list_dir`, `write_file` (gated), `run_bash` (allowlisted + timeout), `finish`. |
| `cogs/cli.py`        | `python -m cogs` REPL and `--replay <cassette>` mode. |

---

## Quickstart

### 1. Offline replay demo (no credentials, no network)

The fastest way to see the loop work. It drives the full agent — message
assembly, tool execution, permission gating — against a checked-in cassette
using `ReplayProvider`.

```bash
uv venv
uv pip install -e '.[dev]'
uv run python examples/replay_demo.py
```

You'll see the recorded model turns drive real `list_dir` / `read_file`
executions and produce a final answer, all with zero API calls.

### 2. Run against real AWS Bedrock

The user of this project has Bedrock access. On Bedrock, model ids take an
`anthropic.` prefix; the default is `anthropic.claude-opus-4-8` (Opus 4.8, the
most capable model).

```bash
AWS_REGION=us-east-1 uv run python examples/coding_agent.py \
  "List the files in the cogs package and summarize what types.py defines."
```

Configuration is entirely via environment variables:

| Variable        | Default (bedrock)             | Meaning |
|-----------------|-------------------------------|---------|
| `COGS_PROVIDER` | `bedrock`                     | `bedrock` or `anthropic` |
| `COGS_MODEL`    | `anthropic.claude-opus-4-8`   | model id (drop the prefix for `anthropic`) |
| `AWS_REGION`    | `us-east-1`                   | Bedrock region |

### 3. Swap to a direct Anthropic API key (future)

If you later get a direct API key, the **only** change is the provider:

```bash
export COGS_PROVIDER=anthropic   # model default becomes claude-opus-4-8 (no prefix)
export ANTHROPIC_API_KEY=sk-...
uv run python examples/coding_agent.py "..."
```

The agent loop, tools, tracing, and tests are untouched — that is the point of
the provider abstraction.

---

## Development

```bash
uv run ruff check .
uv run pytest
```

Both run fully offline. The suite uses a scripted stub provider and hand-authored
cassettes; nothing hits the network.

---

## Design decisions & intentional non-goals

This is a *tasteful mini-implementation*, scoped deliberately. What's left out
is left out on purpose:

- **No streaming UI.** The loop uses non-streaming `messages.create` at
  `max_tokens=8192`, which is well within HTTP timeout limits and keeps the loop
  readable. A production harness streams tokens and renders incrementally; that
  is a presentation concern layered on top of this same loop.
- **No server-side compaction.** `context.py` hand-rolls a trim-oldest strategy
  to demonstrate the token-budget mechanics and the pairing invariant that *any*
  strategy must respect. Production would additionally use the server-side
  compaction beta (`compact-2026-01-12`), which summarizes rather than drops.
- **Adaptive thinking is opt-in, off by default.** Opus 4.8 supports
  `thinking={"type": "adaptive"}`, but Bedrock regional support varies, so it's
  a config flag (`AnthropicProvider(thinking=True)`) rather than a default.
  Sampling params (`temperature`/`top_p`/`top_k`) are never sent — Opus 4.8
  rejects them.
- **Example-grade tool sandboxing.** `run_bash` uses a command allowlist and a
  timeout and rejects shell metacharacters — enough to be safe as a demo, not a
  real sandbox. Production runs tools in a container with a restricted user,
  filesystem confinement, and egress control.
- **Single-file tools.** Tools are plain functions with a decorator; there's no
  plugin system, no dynamic discovery, no MCP. The `@tool` → `ToolRegistry` path
  is small enough to read in one sitting, which is the goal.
- **Synchronous.** No `async`. The loop is sequential and easy to follow; a real
  harness would parallelize independent tool calls.

Each of these is a place where a production system does more — and where this
repo deliberately stops, so the core stays legible.

---

## The platform

`cogs` is one repo in a five-part **agent platform**. Each owns a single concern,
stands alone, and shares the same spine: a normalized, Bedrock-default provider
seam and deterministic, fully-offline tests.

| Repo | Concern |
|------|---------|
| [`cogs`](https://github.com/mbsdeepak/cogs) | the agent **runtime** — the loop, tool protocol, provider seam, record/replay **← this repo** |
| [`bulkhead`](https://github.com/mbsdeepak/bulkhead) | reliable **serving** — a gateway (retries, circuit breaking, rate limits, caching, failover, budgets) in front of any provider |
| [`loom`](https://github.com/mbsdeepak/loom) | **context** engineering — retrieve, compact, and assemble what goes in the window |
| [`sonar`](https://github.com/mbsdeepak/sonar) | **observability** — reconstruct a run as a cost/latency timeline |
| [`gauntlet`](https://github.com/mbsdeepak/gauntlet) | **evaluation** — hermetic tool-use tasks scored with pass@k + confidence intervals |

How work flows through them:

```
loom ──assemble context──▶ cogs ──model calls──▶ bulkhead ──▶ provider
                            │
              run cassette ─┴──▶ sonar (timeline, cost)
              eval result ─────▶ gauntlet (pass@k)
```

The seams are real, not aspirational: `cogs`, `bulkhead`, and `loom` all speak the
same normalized `Provider`/message types, so a call can flow through all three, and
`sonar` ingests `cogs` cassettes and `gauntlet` results directly. The
[`sonar` README](https://github.com/mbsdeepak/sonar#run-it-combined) has the
one-command combined demo.

---

## License

MIT © 2026 Deepak
