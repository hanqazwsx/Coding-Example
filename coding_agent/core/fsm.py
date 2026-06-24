"""
Stage 2: Finite State Machine
==============================
Defines the agent's reasoning states and a configurable transition table.
States: INIT → THINK → ACT → OBSERVE → REFLECT → DONE

The FSM drives the agent's cognitive cycle:
  INIT    → prepare for a new task
  THINK   → LLM reasons about what to do next
  ACT     → invoke a tool or produce output
  OBSERVE → collect the result of the action
  REFLECT → evaluate whether goal is achieved
  DONE    → terminal state
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class State(Enum):
    """Enumeration of all possible FSM states."""
    INIT = auto()
    THINK = auto()
    ACT = auto()
    OBSERVE = auto()
    REFLECT = auto()
    DONE = auto()

    def __str__(self) -> str:
        return self.name


@dataclass
class Transition:
    """A single transition rule between states."""
    from_state: State
    to_state: State
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    description: str = ""

    def is_valid(self, context: Dict[str, Any]) -> bool:
        """Check if this transition's condition (if any) is satisfied."""
        if self.condition is None:
            return True
        try:
            return self.condition(context)
        except Exception as e:
            logger.warning("Transition condition error (%s -> %s): %s",
                           self.from_state, self.to_state, e)
            return False

    def __repr__(self) -> str:
        return f"{self.from_state.name} -> {self.to_state.name}"


# ── Default transition conditions ──────────────────────────────────

def _always(context: Dict[str, Any]) -> bool:
    return True


def _has_tool_result(context: Dict[str, Any]) -> bool:
    """True when the last action produced a tool result."""
    return context.get("_has_tool_result", False)


def _needs_reflection(context: Dict[str, Any]) -> bool:
    """True when the accumulated experience warrants reflection."""
    return context.get("_loop_count", 0) % 3 == 2


def _is_done(context: Dict[str, Any]) -> bool:
    """True when the agent signals completion."""
    return context.get("_done", False)


def _has_more_work(context: Dict[str, Any]) -> bool:
    """True when there are still pending sub-tasks."""
    return not _is_done(context) and context.get("_loop_count", 0) < 20


# ── State abbreviation map (for logging / display) ─────────────────
STATE_ABBR = {
    State.INIT: "INI",
    State.THINK: "THK",
    State.ACT: "ACT",
    State.OBSERVE: "OBS",
    State.REFLECT: "RFL",
    State.DONE: "DON",
}


class FSM:
    """
    Finite State Machine with a configurable transition table.

    Usage:
        fsm = FSM()
        fsm.transition({"some": "context"})   # returns the new State
        print(fsm.current_state)              # the current state
        fsm.reset()                           # back to INIT
    """

    def __init__(self, initial_state: State = State.INIT):
        self._current: State = initial_state
        self._history: List[State] = []
        self._transitions: List[Transition] = self._default_table()
        self._context: Dict[str, Any] = {}

    # ── Default transition table ─────────────────────────────────

    @staticmethod
    def _default_table() -> List[Transition]:
        """
        Build the canonical cognitive cycle:
          INIT → THINK → ACT → OBSERVE → REFLECT → (THINK or DONE)
        """
        return [
            Transition(State.INIT, State.THINK, _always,
                       "Start: enter thinking phase"),
            Transition(State.THINK, State.ACT, _always,
                       "LLM decided on an action"),
            Transition(State.ACT, State.OBSERVE, _always,
                       "After action, observe the result"),
            Transition(State.OBSERVE, State.REFLECT, _always,
                       "After observation, reflect"),
            Transition(State.REFLECT, State.DONE, _is_done,
                       "Task complete → DONE"),
            Transition(State.REFLECT, State.THINK, _has_more_work,
                       "More work remains → continue thinking"),
        ]

    # ── Core API ─────────────────────────────────────────────────

    @property
    def current_state(self) -> State:
        return self._current

    @property
    def history(self) -> List[State]:
        return list(self._history)

    @property
    def current_state_abbr(self) -> str:
        return STATE_ABBR.get(self._current, "???")

    def reset(self, state: State = State.INIT) -> None:
        """Reset the FSM to the given state (default INIT)."""
        self._current = state
        self._history.clear()
        self._context.clear()
        logger.debug("FSM reset to %s", state)

    def set_transition_table(self, transitions: List[Transition]) -> None:
        """Replace the transition table with a custom one."""
        self._transitions = list(transitions)

    def add_transition(self, transition: Transition) -> None:
        """Append a custom transition rule."""
        self._transitions.append(transition)

    def transition(self, context: Optional[Dict[str, Any]] = None) -> State:
        """
        Evaluate the transition table and move to the next valid state.
        Returns the new State.
        """
        if context is not None:
            self._context.update(context)

        valid_moves = [
            t for t in self._transitions
            if t.from_state == self._current and t.is_valid(self._context)
        ]

        if not valid_moves:
            logger.warning(
                "No valid transitions from %s with context keys=%s",
                self._current, list(self._context.keys())
            )
            # Fallback: stay in current state (no-op)
            return self._current

        # For deterministic behavior: pick the first matching transition.
        chosen = valid_moves[0]
        self._history.append(self._current)
        self._current = chosen.to_state
        logger.debug(
            "FSM: %s → %s  (%s)",
            chosen.from_state.name, chosen.to_state.name, chosen.description,
        )
        return self._current

    def can_transition_to(self, target: State) -> bool:
        """Check if a direct transition to `target` is possible from the
        current state (helper for external control)."""
        return any(
            t.from_state == self._current and t.to_state == target
            for t in self._transitions
        )

    def __repr__(self) -> str:
        return f"FSM(state={self._current.name}, steps={len(self._history)})"
