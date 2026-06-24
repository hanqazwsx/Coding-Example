"""
Stage 8: Input Filter — Injection & Path Traversal Detection
=============================================================
Filters user inputs and tool parameters for:
  - Command injection attempts (;, |, &&, `, $(), etc.)
  - Path traversal attempts (../, ..\\, absolute paths outside project)
  - Suspicious patterns

Provides both detection (returns bool) and sanitization (returns cleaned text).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import re

from coding_agent.config import config

logger = logging.getLogger(__name__)


# ── Patterns ───────────────────────────────────────────────────────

# Command injection patterns
COMMAND_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"[;&|]{2,}"),            # &&, ||, ;;(double)
    re.compile(r"(?<!\|)\|(?!\|)"),      # single pipe (not ||)
    re.compile(r"(?<!;);(?!;)"),         # single semicolon (command separator)
    re.compile(r"`[^`]+`"),             # backtick command substitution
    re.compile(r"\$\([^)]+\)"),         # $() command substitution
    re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*"),  # $VARIABLE (allow only known ones)
    re.compile(r">\s*[^>\s]"),          # output redirection (not >>)
    re.compile(r"<\s*[^<\s]"),          # input redirection
]

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"(\.\./|\.\.\\)"),       # ../
    re.compile(r"\.\.(%[0-9a-fA-F]{2})+"),  # URL-encoded ../
]

# Suspicious tool names
SUSPICIOUS_TOOLS = {
    "rm", "del", "rd", "rmdir", "format", "dd",
    "shutdown", "reboot", "init", "poweroff",
    "mkfs", "mount", "fdisk",
    "sudo", "su", "chmod", "chown",
    "iptables", "ufw",
    "useradd", "usermod", "passwd",
}


class InputFilter:
    """
    Filters and sanitizes user inputs and tool parameters.

    Usage:
        filter = InputFilter()
        is_safe, reason = filter.detect_injection("some input")
        safe_input = filter.sanitize_input("risky input")
    """

    def __init__(self, project_root: Optional[str] = None):
        self._project_root = project_root or config.project_root

    # ── Public API ──────────────────────────────────────────────

    def detect_injection(
        self,
        text: str,
    ) -> Tuple[bool, str]:
        """
        Check if text contains injection attempts.

        Returns:
            (is_safe: bool, reason: str)
            is_safe=False means an injection was detected.
        """
        if not isinstance(text, str) or not text.strip():
            return True, ""

        # Check command injection patterns
        for pattern in COMMAND_INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                logger.warning("Command injection detected: %s in '%s'",
                               match.group(), text[:80])
                return False, f"Command injection pattern: '{match.group()}'"

        # Check for suspicious shell commands
        first_word = text.strip().split()[0].lower() if text.strip() else ""
        if first_word in SUSPICIOUS_TOOLS:
            logger.warning("Suspicious command: %s", first_word)
            return False, f"Suspicious/blocked command: '{first_word}'"

        return True, ""

    def detect_path_traversal(
        self,
        path: str,
    ) -> Tuple[bool, str]:
        """
        Check if a path contains traversal attacks or escapes the project root.

        Returns:
            (is_safe: bool, reason: str)
        """
        if not isinstance(path, str) or not path.strip():
            return True, ""

        # Check traversal patterns
        for pattern in PATH_TRAVERSAL_PATTERNS:
            match = pattern.search(path)
            if match:
                logger.warning("Path traversal detected: %s in '%s'",
                               match.group(), path)
                return False, f"Path traversal pattern: '{match.group()}'"

        # Check if resolved path is within project
        try:
            resolved = os.path.abspath(os.path.join(self._project_root, path))
            # Normalise for comparison
            proj = os.path.normpath(self._project_root)
            resolved_norm = os.path.normpath(resolved)

            if not resolved_norm.startswith(proj + os.sep) and resolved_norm != proj:
                logger.warning("Path escapes project root: %s", resolved)
                return False, f"Path '{path}' resolves outside project root"
        except Exception as e:
            return False, f"Path resolution error: {e}"

        return True, ""

    def detect_tool_param_injection(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Check tool-specific parameters for injection.

        For example:
          - read_file / write_file: check path traversal
          - shell_exec: check command injection

        Args:
            tool_name: Name of the tool being called.
            params: Parameter dict.

        Returns:
            (is_safe: bool, reason: str)
        """
        if tool_name in ("read_file", "write_file"):
            path = params.get("path", "")
            return self.detect_path_traversal(path)

        if tool_name == "shell_exec":
            command = params.get("command", "")
            return self.detect_injection(command)

        return True, ""

    def sanitize_input(self, text: str) -> str:
        """
        Sanitize user input by escaping or removing injection patterns.
        Note: sanitization is limited — always prefer detection + rejection
        for security-critical contexts.

        Args:
            text: Raw input text.

        Returns:
            Sanitized text.
        """
        if not text:
            return text

        # Remove backtick-enclosed commands
        text = re.sub(r"`[^`]+`", "[FILTERED]", text)

        # Remove $() commands
        text = re.sub(r"\$\([^)]+\)", "[FILTERED]", text)

        return text

    @staticmethod
    def validate_tool_name(name: str) -> bool:
        """
        Check if a tool name is in the allowed whitelist and not blacklisted.

        Args:
            name: Tool name to check.

        Returns:
            True if the tool is allowed.
        """
        name_lower = name.lower()

        # Check blacklist first
        if any(blocked.lower() == name_lower for blocked in config.tool_blacklist):
            logger.warning("Blocked tool (blacklist): %s", name)
            return False

        # Check whitelist
        if config.tool_whitelist:
            allowed = name_lower in (w.lower() for w in config.tool_whitelist)
            if not allowed:
                logger.warning("Tool not in whitelist: %s", name)
            return allowed

        return True
