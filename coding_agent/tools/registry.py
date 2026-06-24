"""
Stage 3: Tool Registry
======================
Central registry for all available tools.
Supports registration, unregistration, lookup, and enumeration.

Each tool is stored as a (ToolSpec, callable) pair where the callable
is the actual implementation.

Built-in tools:
  - read_file(path)   : read a file from disk
  - write_file(path, content): write content to a file
  - shell_exec(command): execute a shell command (sandbox-aware)
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

from coding_agent.tools.schema import ToolSpec, ToolParameter

logger = logging.getLogger(__name__)

# Type alias: a registered tool is (spec, implementation)
ToolImpl = Callable[..., Any]
RegisteredTool = Tuple[ToolSpec, ToolImpl]


class ToolRegistry:
    """
    Singleton tool registry. Tools are identified by their unique name.

    Usage:
        registry = ToolRegistry.get_instance()
        registry.register(spec, my_func)
        spec, impl = registry.get("read_file")
        all_tools = registry.list_all()
    """

    _instance: Optional["ToolRegistry"] = None

    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}
        self._register_builtins()

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """Get the global ToolRegistry singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── CRUD ────────────────────────────────────────────────────

    def register(
        self,
        spec: ToolSpec,
        implementation: ToolImpl,
        overwrite: bool = False,
    ) -> None:
        """
        Register a new tool.

        Args:
            spec: The tool's specification (name, description, params).
            implementation: The callable that implements the tool.
            overwrite: If True, replace an existing tool with the same name.

        Raises:
            ValueError: If a tool with this name already exists and
                        overwrite is False.
        """
        if spec.name in self._tools and not overwrite:
            raise ValueError(
                f"Tool '{spec.name}' is already registered. "
                "Use overwrite=True to replace."
            )
        self._tools[spec.name] = (spec, implementation)
        logger.info("Registered tool: %s", spec.name)

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        if name in self._tools:
            del self._tools[name]
            logger.info("Unregistered tool: %s", name)
        else:
            logger.warning("Attempted to unregister unknown tool: %s", name)

    def get(self, name: str) -> Optional[RegisteredTool]:
        """Look up a tool by name. Returns (spec, impl) or None."""
        return self._tools.get(name)

    def get_spec(self, name: str) -> Optional[ToolSpec]:
        """Get only the spec for a tool by name."""
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def get_implementation(self, name: str) -> Optional[ToolImpl]:
        """Get only the callable for a tool by name."""
        entry = self._tools.get(name)
        return entry[1] if entry else None

    def list_all(self) -> List[Tuple[str, str]]:
        """List all registered tools as (name, description) pairs."""
        return [(name, spec.description) for name, (spec, _) in self._tools.items()]

    def list_specs(self) -> List[ToolSpec]:
        """Return all tool specs (for LLM function binding)."""
        return [spec for spec, _ in self._tools.values()]

    def tool_exists(self, name: str) -> bool:
        """Check if a tool name is registered."""
        return name in self._tools

    def clear(self) -> None:
        """Unregister all tools (builtins included)."""
        self._tools.clear()
        logger.info("Tool registry cleared.")

    # ── Built-in tools ──────────────────────────────────────────

    def _register_builtins(self) -> None:
        """Register the three built-in example tools."""

        # read_file
        self._tools["read_file"] = (
            ToolSpec(
                name="read_file",
                description="Read the contents of a file from disk.",
                parameters=[
                    ToolParameter(
                        name="path",
                        type="string",
                        description="Absolute or relative path to the file.",
                        required=True,
                    ),
                ],
                category="filesystem",
                tags=["read", "file"],
            ),
            self._builtin_read_file,
        )

        # write_file
        self._tools["write_file"] = (
            ToolSpec(
                name="write_file",
                description="Write content to a file (overwrites existing content).",
                parameters=[
                    ToolParameter(
                        name="path",
                        type="string",
                        description="Absolute or relative path to the file.",
                        required=True,
                    ),
                    ToolParameter(
                        name="content",
                        type="string",
                        description="The content to write.",
                        required=True,
                    ),
                ],
                category="filesystem",
                tags=["write", "file"],
            ),
            self._builtin_write_file,
        )

        # shell_exec
        self._tools["shell_exec"] = (
            ToolSpec(
                name="shell_exec",
                description="Execute a shell command and return its output. "
                            "Only non-interactive commands allowed.",
                parameters=[
                    ToolParameter(
                        name="command",
                        type="string",
                        description="The shell command to execute.",
                        required=True,
                    ),
                ],
                category="system",
                tags=["shell", "command"],
                safe_for_sandbox=False,
            ),
            self._builtin_shell_exec,
        )

        logger.info("Registered 3 built-in tools.")

    # ── Built-in implementations ────────────────────────────────

    @staticmethod
    def _builtin_read_file(path: str) -> str:
        """Read a file and return its content."""
        import os
        full_path = os.path.abspath(path)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    @staticmethod
    def _builtin_write_file(path: str, content: str) -> str:
        """Write content to a file, creating directories if needed."""
        import os
        full_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to {full_path}"

    @staticmethod
    def _builtin_shell_exec(command: str) -> str:
        """
        Execute a shell command. Note: In production this should be
        routed through the security sandbox (Stage 8).
        """
        import subprocess
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        return output.strip()
