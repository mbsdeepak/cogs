"""A small set of safe, dependency-free example tools.

These illustrate the two tool categories a coding agent needs: read-only
inspection (``read_file``, ``list_dir``) that defaults to ``ALLOW``, and
side-effecting actions (``write_file``, ``run_bash``) that a policy should gate
behind ``ASK``.

The sandboxing here is **example-grade**, not production-grade. ``run_bash``
uses a command allowlist and a timeout; a real harness would run in a container
with a restricted user, filesystem confinement, and network egress control. The
allowlist here rejects shell metacharacters and only permits a handful of
read-only commands — enough to be safe as a demo, not enough to be a sandbox.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from .tools import ToolError, ToolRegistry, tool

# Commands ``run_bash`` will execute. Deliberately read-only and small.
_BASH_ALLOWLIST = frozenset({"ls", "cat", "echo", "pwd", "wc", "head", "tail"})
# Shell metacharacters that could chain/redirect commands past the allowlist.
_SHELL_METACHARS = frozenset(";&|`$><\n()")
_BASH_TIMEOUT_SECONDS = 10
# Cap file reads so a huge file cannot blow up the context window.
_MAX_READ_BYTES = 64_000


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file and return its contents.

    Args:
        path: Path to the file to read.
    """
    p = Path(path)
    if not p.is_file():
        raise ToolError(f"not a file: {path}")
    data = p.read_bytes()[:_MAX_READ_BYTES]
    text = data.decode("utf-8", errors="replace")
    if p.stat().st_size > _MAX_READ_BYTES:
        text += f"\n... [truncated at {_MAX_READ_BYTES} bytes]"
    return text


@tool
def list_dir(path: str = ".") -> str:
    """List the entries in a directory, one per line.

    Args:
        path: Directory to list. Defaults to the current directory.
    """
    p = Path(path)
    if not p.is_dir():
        raise ToolError(f"not a directory: {path}")
    entries = sorted(
        f"{child.name}/" if child.is_dir() else child.name for child in p.iterdir()
    )
    return "\n".join(entries) if entries else "(empty)"


@tool
def write_file(path: str, content: str) -> str:
    """Write text to a file, creating parent directories as needed.

    Args:
        path: Destination file path.
        content: Text to write.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {path}"


@tool
def run_bash(command: str) -> str:
    """Run a whitelisted shell command and return combined output.

    Only a small set of read-only commands is permitted, and shell
    metacharacters are rejected. This is example-grade sandboxing, not a real
    sandbox.

    Args:
        command: The command line to run (e.g. "ls -la").
    """
    if any(ch in _SHELL_METACHARS for ch in command):
        raise ToolError("command contains disallowed shell metacharacters")
    parts = shlex.split(command)
    if not parts:
        raise ToolError("empty command")
    if parts[0] not in _BASH_ALLOWLIST:
        raise ToolError(
            f"command {parts[0]!r} is not allowlisted; "
            f"allowed: {sorted(_BASH_ALLOWLIST)}"
        )
    try:
        proc = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=_BASH_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"command timed out after {_BASH_TIMEOUT_SECONDS}s") from exc
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return f"[exit {proc.returncode}]\n{output}"
    return output or "(no output)"


@tool
def finish(answer: str) -> str:
    """Signal completion with a final answer.

    Useful when you want an explicit terminal tool rather than relying on the
    model to stop on its own.

    Args:
        answer: The final answer to return to the user.
    """
    return answer


def default_registry() -> ToolRegistry:
    """A registry pre-populated with all builtin tools."""
    registry = ToolRegistry()
    for func in (read_file, list_dir, write_file, run_bash, finish):
        registry.register(func)
    return registry


__all__ = [
    "read_file",
    "list_dir",
    "write_file",
    "run_bash",
    "finish",
    "default_registry",
]
