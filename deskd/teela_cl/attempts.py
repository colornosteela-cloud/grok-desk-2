"""Short-term causal memory: recent goal attempts awaiting user feedback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .records import AttemptRecord, _now


class AttemptStore:
    def __init__(self, root: Path | None = None, *, maxlen: int = 32) -> None:
        self.root = Path(root) if root is not None else None
        self.path = (self.root / "attempts.jsonl") if self.root is not None else None
        self.maxlen = int(maxlen)
        self._rows: list[AttemptRecord] = []
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(raw, dict):
                self._rows.append(AttemptRecord.from_dict(raw))
        self._rows = self._rows[-self.maxlen :]

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = "\n".join(json.dumps(r.to_dict()) for r in self._rows[-self.maxlen :]) + (
            "\n" if self._rows else ""
        )
        self.path.write_text(blob, encoding="utf-8")

    def all(self) -> list[AttemptRecord]:
        return list(self._rows)

    def latest(self, *, awaiting: bool | None = None) -> AttemptRecord | None:
        rows = list(reversed(self._rows))
        for rec in rows:
            if awaiting is None or rec.awaiting_feedback is awaiting:
                return rec
        return None

    def get(self, attempt_id: str) -> AttemptRecord | None:
        for rec in reversed(self._rows):
            if rec.attempt_id == attempt_id:
                return rec
        return None

    def record(
        self,
        *,
        goal: str,
        request: str = "",
        skill: str | None = None,
        plan: list[dict[str, Any]] | None = None,
        observations: dict[str, Any] | None = None,
        outcome: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
        goal_id: str | None = None,
        awaiting_feedback: bool = True,
    ) -> AttemptRecord:
        rec = AttemptRecord(
            attempt_id=f"attempt_{uuid4().hex[:8]}",
            goal_id=goal_id or f"goal_{uuid4().hex[:8]}",
            goal=goal,
            skill=skill,
            plan=list(plan or []),
            started_at=_now(),
            completed_at=_now(),
            observations=dict(observations or {}),
            outcome=dict(outcome or {}),
            evaluation=dict(evaluation or {}),
            awaiting_feedback=awaiting_feedback,
            request=request or goal,
        )
        self._rows.append(rec)
        self._rows = self._rows[-self.maxlen :]
        self.save()
        return rec

    def update(self, rec: AttemptRecord) -> AttemptRecord:
        for i, row in enumerate(self._rows):
            if row.attempt_id == rec.attempt_id:
                self._rows[i] = rec
                self.save()
                return rec
        self._rows.append(rec)
        self.save()
        return rec

    def negatives_for(self, *, skill: str | None = None, goal: str | None = None, goal_id: str | None = None) -> int:
        n = 0
        for rec in self._rows:
            if rec.error_class in {None, "validation"}:
                continue
            if rec.awaiting_feedback and rec.error_class is None:
                continue
            if goal_id:
                if rec.goal_id == goal_id and rec.error_class not in {None, "validation"}:
                    n += 1
                continue
            if skill and rec.skill == skill and rec.error_class in {"skill", "system", "planning"}:
                n += 1
            elif goal and rec.goal == goal and rec.error_class not in {None, "validation"}:
                n += 1
        return n
