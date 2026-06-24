"""
Stage 3: Tool Executor
======================
Unified executor that validates parameters, invokes the tool, and
returns a standardised ToolResult with timing and error handling.

Supports pre/post hooks for security filtering, tracing, and audit.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import time
import inspect

from pydantic import ValidationError

from coding_agent.tools.schema import ToolResult, ToolSpec
from coding_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Type for pre/post hooks
HookFn = Callable[[str, Dict[str, Any]], None]


class ToolExecutor:
    """
    Executes tools by name with parameter validation and error handling.

    Args:
        registry: ToolRegistry instance (defaults to singleton).
        pre_hooks: Called before execution with (tool_name, params).
        post_hooks: Called after execution with (tool_name, params, result).
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        pre_hooks: Optional[List[HookFn]] = None,
        post_hooks: Optional[List[HookFn]] = None,
    ):
        self._registry = registry or ToolRegistry.get_instance()
        self._pre_hooks = pre_hooks or []
        self._post_hooks = post_hooks or []

    def add_pre_hook(self, hook: HookFn) -> None:
        """Add a pre-execution hook."""
        self._pre_hooks.append(hook)

    def add_post_hook(self, hook: HookFn) -> None:
        """Add a post-execution hook."""
        self._post_hooks.append(hook)

    def execute(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name with the given parameters.

        Steps:
          1. Look up the tool in the registry.
          2. Validate parameters (type coercion + required fields).
          3. Run pre-hooks.
          4. Execute with timing.
          5. Run post-hooks.
          6. Return ToolResult.

        Args:
            tool_name: The registered tool name.
            params: Dict of parameter names to values.

        Returns:
            ToolResult with success/failure, result/error, and duration.
        """
        start_time = time.perf_counter()

        # 1. Lookup
        entry = self._registry.get(tool_name)
        if entry is None:
            return ToolResult(
                success=False,
                error=f"Unknown tool: '{tool_name}'. "
                      f"Available: {[n for n, _ in self._registry.list_all()]}",
                duration_ms=0,
                tool_name=tool_name,
            )

        spec, implementation = entry

        # 2. Validate parameters
        try:
            validated_params = self._validate_params(spec, params)
        except (ValidationError, ValueError) as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=False,
                error=f"Parameter validation error: {e}",
                duration_ms=elapsed,
                tool_name=tool_name,
            )

        # 3. Pre-hooks
        self._run_hooks(self._pre_hooks, tool_name, validated_params)

        # 4. Execute
        try:
            # Inspect the function signature to pass only accepted kwargs
            sig = inspect.signature(implementation)
            filtered_params = {
                k: v for k, v in validated_params.items()
                if k in sig.parameters
            }
            result_data = implementation(**filtered_params)
            elapsed = (time.perf_counter() - start_time) * 1000
            tr = ToolResult(
                success=True,
                result=result_data,
                error="",
                duration_ms=round(elapsed, 2),
                tool_name=tool_name,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            tr = ToolResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
                duration_ms=round(elapsed, 2),
                tool_name=tool_name,
            )
            logger.warning("Tool '%s' failed: %s", tool_name, e)

        # 5. Post-hooks
        self._run_hooks(self._post_hooks, tool_name, validated_params)

        return tr

    def execute_multi(
        self,
        calls: List[Tuple[str, Dict[str, Any]]],
    ) -> List[ToolResult]:
        """
        Execute multiple tools in sequence.

        Args:
            calls: List of (tool_name, params) tuples.

        Returns:
            List of ToolResult in the same order.
        """
        return [self.execute(name, params) for name, params in calls]

    # ── Internal ────────────────────────────────────────────────

    @staticmethod
    def _validate_params(spec: ToolSpec, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and coerce parameters against the spec.
        - Checks required fields are present
        - Coerces types where possible
        """
        validated = dict(params)

        for param in spec.parameters:
            # Check required
            if param.name not in validated:
                if param.required:
                    raise ValueError(f"Missing required parameter: '{param.name}'")
                # Optional: set default
                if param.default is not None:
                    validated[param.name] = param.default
                continue

            # Type coercion (basic)
            value = validated[param.name]
            if param.type == "string" and not isinstance(value, str):
                validated[param.name] = str(value)
            elif param.type == "integer" and not isinstance(value, int):
                validated[param.name] = int(value)
            elif param.type == "number" and not isinstance(value, (int, float)):
                validated[param.name] = float(value)
            elif param.type == "boolean" and not isinstance(value, bool):
                validated[param.name] = bool(value)

        return validated

    @staticmethod
    def _run_hooks(
        hooks: List[HookFn],
        tool_name: str,
        params: Dict[str, Any],
    ) -> None:
        """Run a list of hooks, catching and logging errors."""
        for hook in hooks:
            try:
                hook(tool_name, params)
            except Exception as e:
                logger.warning("Hook error for '%s': %s", tool_name, e)
