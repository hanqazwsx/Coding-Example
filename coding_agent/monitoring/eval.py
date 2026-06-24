"""
Stage 9: Task Effect Evaluation Script
=======================================
Runs a battery of predefined test tasks through the agent and computes
metrics: tool call success rate, task completion rate, average turns.

This script can be run standalone:
    python -m coding_agent.monitoring.eval

Or imported and used programmatically:
    from coding_agent.monitoring.eval import Evaluator
    evaluator = Evaluator()
    report = evaluator.run()
    print(report)
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import time
import json
import sys
import os

# Add parent to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# ── Predefined test tasks ──────────────────────────────────────────

DEFAULT_TEST_TASKS: List[Dict[str, Any]] = [
    {
        "id": "read_file_test",
        "description": "Read a file from disk",
        "instruction": "Read the file 'config.py' in the current project.",
        "expected_tools": ["read_file"],
        "min_tool_calls": 1,
        "tags": ["basic", "filesystem"],
    },
    {
        "id": "write_file_test",
        "description": "Write content to a file",
        "instruction": "Write 'Hello, world!' to a file called 'test_output.txt'.",
        "expected_tools": ["write_file"],
        "min_tool_calls": 1,
        "tags": ["basic", "filesystem"],
    },
    {
        "id": "check_dir_listing",
        "description": "List files in the project directory",
        "instruction": "List all Python files in the current directory.",
        "expected_tools": ["shell_exec"],
        "min_tool_calls": 1,
        "tags": ["basic", "shell"],
    },
    {
        "id": "complex_read_write",
        "description": "Read a file and write a modified copy",
        "instruction": "Read 'config.py', then write a summary of its contents to 'config_summary.txt'.",
        "expected_tools": ["read_file", "write_file"],
        "min_tool_calls": 2,
        "tags": ["intermediate", "filesystem"],
    },
]


class Evaluator:
    """
    Runs test tasks through the agent pipeline and produces a report.

    Args:
        tasks: List of test task dicts. Defaults to DEFAULT_TEST_TASKS.
        agent_factory: Callable that returns a configured agent for each task.
                       If None, uses a simplified in-process evaluator.
    """

    def __init__(
        self,
        tasks: Optional[List[Dict[str, Any]]] = None,
        agent_factory: Optional[Callable] = None,
    ):
        self._tasks = tasks or DEFAULT_TEST_TASKS
        self._agent_factory = agent_factory
        self._results: List[Dict[str, Any]] = []

    # ── Public API ──────────────────────────────────────────────

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute all test tasks and compute metrics.

        Args:
            verbose: If True, print per-task progress.

        Returns:
            Dict with: summary, per_task_results, metrics
        """
        self._results = []
        start_time = time.perf_counter()

        logger.info("Starting evaluation with %d tasks", len(self._tasks))

        for i, task in enumerate(self._tasks):
            task_id = task.get("id", f"task_{i}")
            if verbose:
                print(f"[{i+1}/{len(self._tasks)}] {task_id}: {task['description']}...")

            result = self._evaluate_single(task)
            self._results.append(result)

            if verbose:
                status = "[OK]" if result["success"] else "[FAIL]"
                print(f"  {status} tools={result['tool_call_count']}, "
                      f"turns={result['turns']}, "
                      f"duration={result['duration_ms']:.0f}ms")

        total_time = (time.perf_counter() - start_time) * 1000
        metrics = self._compute_metrics()

        report = {
            "summary": {
                "total_tasks": len(self._tasks),
                "completed": metrics["completed"],
                "failed": metrics["failed"],
                "total_tool_calls": metrics["total_tool_calls"],
                "total_duration_ms": round(total_time, 2),
            },
            "metrics": metrics,
            "per_task_results": self._results,
        }

        # Print summary
        if verbose:
            print("\n" + "=" * 50)
            print("EVALUATION REPORT")
            print("=" * 50)
            print(f"  Tasks:    {metrics['completed']}/{len(self._tasks)} completed")
            print(f"  Rate:     {metrics['task_completion_rate']:.1%}")
            print(f"  Tool SR:  {metrics['tool_call_success_rate']:.1%}")
            print(f"  Avg turn: {metrics['avg_turns_per_task']:.1f}")
            print(f"  Avg dur:  {metrics['avg_duration_ms']:.0f}ms")
            print(f"  Total:    {total_time:.0f}ms")
            print("=" * 50)

        return report

    def export_report(self, report: Dict[str, Any], filepath: Optional[str] = None) -> str:
        """Export the evaluation report as JSON."""
        path = filepath or os.path.join(
            os.path.dirname(__file__), "eval_report.json"
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        logger.info("Report exported to %s", path)
        return path

    # ── Internal ────────────────────────────────────────────────

    def _evaluate_single(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single task through the agent.

        Args:
            task: Task definition dict.

        Returns:
            Result dict with: id, success, tool_call_count, turns,
            duration_ms, tools_used, error
        """
        task_start = time.perf_counter()

        if self._agent_factory:
            # Use external agent
            try:
                agent = self._agent_factory()
                result = agent(task["instruction"])
                duration = (time.perf_counter() - task_start) * 1000
                return self._parse_agent_result(task, result, duration)
            except Exception as e:
                duration = (time.perf_counter() - task_start) * 1000
                return {
                    "id": task["id"],
                    "success": False,
                    "error": str(e),
                    "tool_call_count": 0,
                    "tool_success_count": 0,
                    "turns": 0,
                    "duration_ms": round(duration, 2),
                    "tools_used": [],
                }

        # Use simplified in-process evaluator:
        # simulate tool calls based on what the task expects
        return self._evaluate_simple(task, task_start)

    def _evaluate_simple(
        self,
        task: Dict[str, Any],
        start: float,
    ) -> Dict[str, Any]:
        """
        Simplified evaluation that exercises the tool registry directly
        and verifies the expected tools can be called successfully.
        """
        from coding_agent.tools.registry import ToolRegistry
        from coding_agent.tools.executor import ToolExecutor

        registry = ToolRegistry.get_instance()
        executor = ToolExecutor(registry)

        expected_tools = task.get("expected_tools", [])
        tool_results = []
        success = True
        error = ""

        for tool_name in expected_tools:
            # Build dummy params
            params = {}
            spec = registry.get_spec(tool_name)
            if spec:
                for p in spec.parameters:
                    if p.name == "path":
                        # Use a safe test path
                        import os
                        test_dir = os.path.dirname(os.path.dirname(__file__))
                        if tool_name == "read_file":
                            params["path"] = os.path.join(test_dir, "config.py")
                        else:
                            params["path"] = os.path.join(test_dir, "_eval_test_output.txt")
                    elif p.name == "content":
                        params["content"] = "eval test content"
                    elif p.name == "command":
                        params["command"] = "ls"

            tr = executor.execute(tool_name, params)
            tool_results.append({
                "tool": tool_name,
                "success": tr.success,
                "error": tr.error,
            })
            if not tr.success:
                success = False
                error = f"Tool '{tool_name}' failed: {tr.error}"
                break

        duration = (time.perf_counter() - start) * 1000

        return {
            "id": task["id"],
            "success": success,
            "error": error,
            "tool_call_count": len(expected_tools),
            "tool_success_count": sum(1 for r in tool_results if r["success"]),
            "turns": len(expected_tools) + 2,  # approximate: think + act per tool + reflect
            "duration_ms": round(duration, 2),
            "tools_used": expected_tools,
            "tool_details": tool_results,
        }

    @staticmethod
    def _parse_agent_result(
        task: Dict[str, Any],
        agent_result: Any,
        duration_ms: float,
    ) -> Dict[str, Any]:
        """Parse a full agent result into evaluation format."""
        if isinstance(agent_result, dict):
            tool_results = agent_result.get("tool_results", [])
            return {
                "id": task.get("id", "unknown"),
                "success": agent_result.get("final_state") == "DONE",
                "error": "",
                "tool_call_count": len(tool_results),
                "tool_success_count": sum(
                    1 for r in tool_results if isinstance(r, dict) and r.get("success")
                ),
                "turns": agent_result.get("iterations", 0),
                "duration_ms": round(duration_ms, 2),
                "tools_used": list(set(
                    r.get("tool_name", "") for r in tool_results
                    if isinstance(r, dict)
                )),
            }

        return {
            "id": task.get("id", "unknown"),
            "success": bool(agent_result),
            "error": "",
            "tool_call_count": 0,
            "turns": 1,
            "duration_ms": round(duration_ms, 2),
            "tools_used": [],
        }

    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics from all task results."""
        if not self._results:
            return {
                "completed": 0,
                "failed": 0,
                "task_completion_rate": 0.0,
                "total_tool_calls": 0,
                "tool_call_success_rate": 0.0,
                "avg_turns_per_task": 0.0,
                "avg_duration_ms": 0.0,
            }

        completed = sum(1 for r in self._results if r["success"])
        total_tool_calls = sum(r["tool_call_count"] for r in self._results)
        total_tool_success = sum(r["tool_success_count"] for r in self._results)
        total_turns = sum(r["turns"] for r in self._results)
        total_duration = sum(r["duration_ms"] for r in self._results)

        return {
            "completed": completed,
            "failed": len(self._results) - completed,
            "task_completion_rate": round(
                completed / len(self._results), 4
            ) if self._results else 0.0,
            "total_tool_calls": total_tool_calls,
            "tool_call_success_rate": round(
                total_tool_success / total_tool_calls, 4
            ) if total_tool_calls else 0.0,
            "avg_turns_per_task": round(
                total_turns / len(self._results), 2
            ) if self._results else 0.0,
            "avg_duration_ms": round(
                total_duration / len(self._results), 2
            ) if self._results else 0.0,
        }


# ── Standalone entry point ─────────────────────────────────────────

def main():
    """Run evaluation from command line."""
    from coding_agent.config import setup_logging
    setup_logging()
    print("=" * 50)
    print("coding_agent Evaluation Suite")
    print("=" * 50)

    evaluator = Evaluator()
    report = evaluator.run(verbose=True)

    export_path = evaluator.export_report(report)
    print(f"\nReport exported to: {export_path}")

    # Return exit code based on success
    metrics = report["metrics"]
    if metrics["task_completion_rate"] < 0.5:
        print("WARNING: Task completion rate below 50%")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
