"""
Stage 7: Worker Agent
======================
A self-contained worker agent that can execute a subtask independently.
Each worker has its own FSM, QueryLoop, and ShortTermMemory.

Workers are spawned by the Orchestrator for parallel sub-task execution.
They communicate results back through a shared queue or callback.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import logging
import threading
import uuid

from coding_agent.config import config
from coding_agent.core.fsm import FSM
from coding_agent.core.query_loop import QueryLoop
from coding_agent.tools.executor import ToolExecutor
from coding_agent.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)


# Shared result queue (used across threads)
_worker_results: Dict[str, Any] = {}
_results_lock = threading.Lock()


class WorkerAgent:
    """
    A lightweight worker agent that executes a single subtask.

    Args:
        worker_id: Unique ID (auto-generated if not provided).
        tool_executor: Shared ToolExecutor instance.
        result_callback: Called with (worker_id, result) when done.
        max_iterations: Max FSM iterations for this worker.
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        tool_executor: Optional[ToolExecutor] = None,
        result_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        max_iterations: int = config.worker_max_iterations,
    ):
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:6]}"
        self._tool_executor = tool_executor or ToolExecutor()
        self._result_callback = result_callback
        self._max_iterations = max_iterations

        # Each worker has its own FSM, memory, and query loop
        self.fsm = FSM()
        self.memory = ShortTermMemory(maxlen=config.short_term_maxlen)

        self._loop = QueryLoop(
            fsm=self.fsm,
            tool_executor=self._execute_tool,
        )

        # Execution state
        self._status: str = "idle"  # idle | running | done | error
        self._result: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None

    # ── Public API ──────────────────────────────────────────────

    @property
    def status(self) -> str:
        return self._status

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        return self._result

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the given subtask.

        Args:
            task: Dict with keys:
                - instruction (str): what to do
                - context (dict, optional): extra context
                - tools (list, optional): tools available for this task

        Returns:
            Result dict with keys: worker_id, status, result, error, iterations
        """
        self._status = "running"
        logger.info("Worker %s starting task: %s", self.worker_id, str(task)[:80])

        instruction = task.get("instruction", "")
        task_context = task.get("context", {})
        tools = task.get("tools", [])

        # Build system prompt with task context
        system_prompt = (
            f"You are Worker Agent '{self.worker_id}'. "
            f"Your task is: {instruction}\n"
        )
        if task_context:
            from pprint import pformat
            system_prompt += f"\nContext:\n{pformat(task_context, indent=2)}"

        self._loop.set_system_prompt(system_prompt)

        # If we have langchain tools, convert them
        lc_tools = self._convert_to_langchain_tools(tools) if tools else None

        try:
            result = self._loop.run(
                user_input=instruction,
                tools=lc_tools,
            )
            self._result = {
                "worker_id": self.worker_id,
                "status": "done",
                "result": result.get("response", ""),
                "tool_results": result.get("tool_results", []),
                "iterations": result.get("iterations", 0),
                "error": "",
            }
            self._status = "done"

            # Store in global results
            with _results_lock:
                _worker_results[self.worker_id] = self._result

            if self._result_callback:
                self._result_callback(self.worker_id, self._result)

            logger.info("Worker %s completed in %d iterations.",
                        self.worker_id, result.get("iterations", 0))

        except Exception as e:
            self._status = "error"
            self._error = str(e)
            self._result = {
                "worker_id": self.worker_id,
                "status": "error",
                "result": "",
                "tool_results": [],
                "iterations": 0,
                "error": str(e),
            }
            logger.error("Worker %s failed: %s", self.worker_id, e)

        return self._result

    # ── Internal ────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and record in memory."""
        tr = self._tool_executor.execute(tool_name, params)
        self.memory.add(
            role="tool",
            content=f"Tool '{tool_name}': success={tr.success}, "
                    f"result={str(tr.result)[:200]}",
        )
        return tr.to_dict()

    @staticmethod
    def _convert_to_langchain_tools(tool_specs: List[Dict[str, Any]]) -> List:
        """
        Convert tool specs to langchain Tool objects.
        Uses the standard langchain BaseTool.from_function pattern.
        """
        from coding_agent.tools.registry import ToolRegistry
        from langchain_core.tools import Tool

        registry = ToolRegistry.get_instance()
        lc_tools = []

        for spec in tool_specs:
            name = spec.get("name", "")
            entry = registry.get(name)
            if entry is None:
                continue
            spec_obj, impl = entry

            lc_tools.append(Tool(
                name=name,
                func=impl,
                description=spec_obj.description,
            ))

        return lc_tools
