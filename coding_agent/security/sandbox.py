"""
Stage 8: Shell Sandbox
======================
Restricts shell command execution to a safe subset of commands.
Enforces:
  - Command allowlist (only approved commands can run)
  - No network connections (blocks curl, wget, nc, ssh, etc.)
  - Working directory locked to project root
  - Timeout to prevent runaway processes

All shell_exec calls should be routed through this sandbox.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import re
import subprocess
import shlex

from coding_agent.config import config
from coding_agent.tools.schema import ToolResult

logger = logging.getLogger(__name__)


class ShellSandbox:
    """
    Safe shell execution environment.

    Usage:
        sandbox = ShellSandbox()
        result = sandbox.execute("ls -la")
        # → ToolResult(success=True, result="...", ...)
    """

    def __init__(
        self,
        allowed_commands: Optional[List[str]] = None,
        blocked_patterns: Optional[List[str]] = None,
        project_root: Optional[str] = None,
        timeout: int = 30,
    ):
        self._allowed_commands = [
            cmd.lower() for cmd in (allowed_commands or config.shell_allowed_commands)
        ]
        self._blocked_patterns = blocked_patterns or config.shell_blocked_patterns
        self._project_root = project_root or config.project_root
        self._timeout = timeout

    # ── Public API ──────────────────────────────────────────────

    def execute(self, command: str) -> ToolResult:
        """
        Execute a shell command within the sandbox constraints.

        Args:
            command: Shell command string.

        Returns:
            ToolResult with stdout/stderr or error.
        """
        import time
        start = time.perf_counter()

        # 1. Validate command
        is_valid, reason = self._validate(command)
        if not is_valid:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=f"Sandbox rejected command: {reason}",
                duration_ms=round(elapsed, 2),
                tool_name="shell_exec",
            )

        # 2. Execute
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=self._project_root,
            )
            elapsed = (time.perf_counter() - start) * 1000

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"

            return ToolResult(
                success=(result.returncode == 0),
                result=output.strip() if output.strip() else "(no output)",
                error="" if result.returncode == 0 else f"Exit code {result.returncode}",
                duration_ms=round(elapsed, 2),
                tool_name="shell_exec",
            )

        except subprocess.TimeoutExpired:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=f"Command timed out after {self._timeout}s",
                duration_ms=round(elapsed, 2),
                tool_name="shell_exec",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                success=False,
                error=f"Execution error: {e}",
                duration_ms=round(elapsed, 2),
                tool_name="shell_exec",
            )

    # ── Internal Validation ─────────────────────────────────────

    def _validate(self, command: str) -> Tuple[bool, str]:
        """
        Validate a command against sandbox rules.

        Returns:
            (is_valid: bool, reason: str)
        """
        if not command or not command.strip():
            return False, "Empty command"

        cmd_stripped = command.strip()

        # Check blocked patterns
        for pattern in self._blocked_patterns:
            if pattern.lower() in cmd_stripped.lower():
                return False, f"Pattern '{pattern}' is blocked"

        # Extract the base command (first word, handling pipes/chains)
        base_cmd = self._extract_base_command(cmd_stripped)
        if base_cmd is None:
            return False, "Could not parse base command"

        # Check against allowlist
        if base_cmd.lower() not in self._allowed_commands:
            return False, (
                f"Command '{base_cmd}' is not in the allowed list. "
                f"Allowed: {', '.join(self._allowed_commands)}"
            )

        # Check for dangerous shell metacharacters in complex expressions
        # Validate ALL segments separated by |, ;, &&, ||
        for sep in ("|", ";", "&&", "||"):
            if sep in cmd_stripped:
                segments = cmd_stripped.split(sep)
                for seg in segments:
                    seg = seg.strip()
                    if seg:
                        seg_cmd = self._extract_base_command(seg)
                        if seg_cmd and seg_cmd.lower() not in self._allowed_commands:
                            return False, (
                                f"Command '{seg_cmd}' in '{sep}'-chain is not allowed"
                            )

        return True, ""

    @staticmethod
    def _extract_base_command(command: str) -> Optional[str]:
        """
        Extract the base command from a shell expression.
        Handles: cmd, cmd args, cmd | cmd, cmd; cmd
        """
        # Remove leading/trailing whitespace
        cmd = command.strip()

        # If there's a pipe or semicolon, take the first segment
        for sep in ("|", ";", "&&", "||"):
            if sep in cmd:
                cmd = cmd.split(sep)[0].strip()

        # Take the first word
        parts = cmd.split()
        return parts[0] if parts else None
