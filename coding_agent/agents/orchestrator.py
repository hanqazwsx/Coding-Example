"""
Stage 7: Central Orchestrator — Master Agent
=============================================
Decomposes high-level tasks into subtasks and dispatches them to
worker agents for parallel execution using ThreadPoolExecutor.

Supports:
  - Task decomposition: Uses LLM to split a complex task into subtasks.
  - Fork-join: Spawns workers in parallel, collects results.
  - Resource limiting: Caps concurrent workers for notebook environments.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

from coding_agent.config import config
from coding_agent.agents.worker import WorkerAgent
from coding_agent.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Master agent that orchestrates parallel worker execution.

    Args:
        tool_executor: Shared ToolExecutor instance.
        max_workers: Max concurrent workers (default 4 for notebook safety).
        result_callback: Called with (worker_id, result) for each completion.
    """

    def __init__(
        self,
        tool_executor: Optional[ToolExecutor] = None,
        max_workers: int = config.orchestrator_max_workers,
        result_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self._tool_executor = tool_executor or ToolExecutor()
        self._max_workers = max_workers
        self._result_callback = result_callback

        # Track spawned workers
        self._workers: Dict[str, WorkerAgent] = {}
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

    # ── Public API ──────────────────────────────────────────────

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a high-level task by decomposing it and running subtasks
        in parallel.

        Args:
            task: Dict with keys:
                - instruction (str): High-level task description.
                - context (dict, optional): Shared context for all subtasks.
                - tools (list, optional): Available tools.
                - subtasks (list, optional): Pre-defined subtask list.
                    If not provided, the LLM will decompose.

        Returns:
            Aggregated result dict: {
                status, subtasks, results (list), errors (list),
                total_workers, total_time_s
            }
        """
        import time
        start_time = time.time()

        instruction = task.get("instruction", "")
        context = task.get("context", {})
        tools = task.get("tools", [])

        # 1. Decompose into subtasks
        subtasks = task.get("subtasks")
        if not subtasks:
            subtasks = self._decompose_task(instruction, context)
            logger.info(
                "Task decomposed into %d subtasks.",
                len(subtasks),
            )

        if not subtasks:
            return {
                "status": "error",
                "error": "Task decomposition produced no subtasks.",
                "results": [],
                "total_workers": 0,
                "total_time_s": round(time.time() - start_time, 2),
            }

        # 2. Filter context for each subtask
        enriched_subtasks = []
        for i, st in enumerate(subtasks):
            enriched_subtasks.append({
                "instruction": st if isinstance(st, str) else st.get("instruction", str(st)),
                "context": {**context, "subtask_index": i},
                "tools": tools,
            })

        # 3. Fork-join execution
        results = self._run_parallel(enriched_subtasks)

        # 4. Aggregate
        total_time = round(time.time() - start_time, 2)
        errors = [r for r in results if r.get("status") == "error"]

        aggregated = {
            "status": "completed" if not errors else "partial",
            "subtasks": subtasks,
            "results": results,
            "errors": errors,
            "total_workers": len(results),
            "total_time_s": total_time,
        }

        logger.info(
            "Orchestrator finished: %d/%d workers done, %d errors, %.2fs",
            len(results) - len(errors),
            len(results),
            len(errors),
            total_time,
        )

        return aggregated

    def run_with_callback(
        self,
        task: Dict[str, Any],
        on_progress: Callable[[str, Dict[str, Any]], None],
    ) -> Dict[str, Any]:
        """
        Run with a progress callback invoked as each worker completes.

        Args:
            task: Same as run().
            on_progress: Called with (worker_id, result) as each finishes.
        """
        original_cb = self._result_callback
        self._result_callback = lambda wid, res: (
            on_progress(wid, res) or (original_cb and original_cb(wid, res))
        )
        try:
            return self.run(task)
        finally:
            self._result_callback = original_cb

    def get_workers(self) -> List[Dict[str, Any]]:
        """Return status of all workers."""
        return [
            {
                "worker_id": w.worker_id,
                "status": w.status,
                "has_result": w.result is not None,
            }
            for w in self._workers.values()
        ]

    def shutdown(self) -> None:
        """Shut down the thread pool."""
        self._executor.shutdown(wait=False)
        self._workers.clear()
        logger.info("Orchestrator shut down.")

    # ── Internal ────────────────────────────────────────────────

    def _decompose_task(
        self,
        instruction: str,
        context: Dict[str, Any],
    ) -> List[str]:
        """
        Use LLM to decompose a high-level task into parallel subtasks.

        Args:
            instruction: The high-level task.
            context: Extra context.

        Returns:
            List of subtask instruction strings.
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatOpenAI(
                model=config.deepseek_model,
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
                temperature=0.2,
                max_tokens=1024,
                timeout=30,
            )

            system_msg = SystemMessage(
                content=(
                    "You are a task decomposition assistant. Given a high-level "
                    "task, break it down into 2-4 concrete, independently executable "
                    "subtasks. Each subtask should be a clear instruction that can "
                    "be given to a worker agent.\n\n"
                    "Output ONLY a JSON list of strings, no other text:\n"
                    '["subtask 1", "subtask 2", ...]'
                )
            )
            human_msg = HumanMessage(
                content=f"Task: {instruction}\n\nContext: {json.dumps(context, default=str)}"
            )

            response = llm.invoke([system_msg, human_msg])
            text = (response.content or "").strip()

            # Extract JSON list
            if "[" in text and "]" in text:
                json_str = text[text.index("["):text.rindex("]") + 1]
                subtasks = json.loads(json_str)
                if isinstance(subtasks, list) and all(isinstance(s, str) for s in subtasks):
                    return subtasks

            # Fallback
            return [instruction]

        except Exception as e:
            logger.warning("Task decomposition failed: %s", e)
            return [instruction]

    def _run_parallel(
        self,
        subtasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run subtasks in parallel using ThreadPoolExecutor.

        Args:
            subtasks: List of task dicts.

        Returns:
            List of result dicts (order not guaranteed to match input).
        """
        futures: List[Future] = []
        result_map: Dict[Future, str] = {}

        for st in subtasks:
            worker = WorkerAgent(
                tool_executor=self._tool_executor,
                result_callback=self._result_callback,
            )
            self._workers[worker.worker_id] = worker

            future = self._executor.submit(worker.execute, st)
            futures.append(future)
            result_map[future] = worker.worker_id

        # Collect results as they complete
        results = []
        for future in as_completed(futures):
            worker_id = result_map[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error("Worker %s thread failed: %s", worker_id, e)
                results.append({
                    "worker_id": worker_id,
                    "status": "error",
                    "error": str(e),
                    "result": "",
                })

        return results
