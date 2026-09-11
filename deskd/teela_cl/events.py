"""Structured events for capability, learning, and RSI. Not model prose."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable


EVENT_TYPES = (
    "goal_interpreted",
    "capability_assessed",
    "skill_resolved",
    "skill_composition_started",
    "learning_started",
    "learning_attempt",
    "simulation_result",
    "execution_result",
    "outcome_evaluated",
    "skill_compiled",
    "skill_validated",
    "skill_persisted",
    "attempt_recorded",
    "feedback_interpreted",
    "correction_diagnosed",
    "correction_retried",
    "skill_updated",
    "feedback_asked",
    "improvement_observed",
    "improvement_proposed",
    "candidate_built",
    "candidate_evaluated",
    "candidate_rejected",
    "candidate_passed",
    "candidate_promoted",
    "rollback_performed",
)


class EventLog:
    def __init__(self, sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self._sink = sink

    def emit(self, kind: str, **payload: Any) -> dict[str, Any]:
        if kind not in EVENT_TYPES:
            raise ValueError(f"unknown event type {kind!r}")
        ev = {
            "type": kind,
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **payload,
        }
        self.events.append(ev)
        if self._sink is not None:
            try:
                self._sink(ev)
            except Exception:
                pass
        return ev

    def kinds(self) -> list[str]:
        return [str(e.get("type") or "") for e in self.events]
