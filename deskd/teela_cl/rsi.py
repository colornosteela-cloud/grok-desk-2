"""Bounded RSI: propose, isolate, evaluate, gate, promote, rollback.

Never silently replaces the active learner. Candidate code cannot drop
protected invariants or the promotion gate itself.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .events import EventLog
from .records import (
    CandidateState,
    ImprovementObservation,
    ImprovementRecord,
    LearnerSnapshot,
    _now,
)

PROTECTED_INVARIANTS = frozenset(
    {
        "emergency_stop",
        "permission_enforcement",
        "sandbox_boundaries",
        "hardware_safety_limits",
        "motor_limits",
        "authentication_policy",
        "rollback_infrastructure",
        "audit_history",
        "evaluation_gates",
        "promotion_rules",
    }
)


def default_snapshot() -> LearnerSnapshot:
    return LearnerSnapshot(
        version="learner_v1",
        retrieval_threshold=0.85,
        max_attempts=4,
        trajectory_weight=0.0,
        replay_penalty=1.0,
        invariants=sorted(PROTECTED_INVARIANTS),
    )


# Cost metrics: fewer is better (journal example: 8 trials → 3 is an improvement).
_LOWER_IS_BETTER = frozenset(
    {
        "attempts",
        "failures",
        "replays",
        "replay_count",
        "mean_plan_length",
        "latency",
        "cost",
        "error_rate",
    }
)


def metrics_candidate_better(baseline: dict[str, float], candidate: dict[str, float]) -> bool:
    """True if candidate improves at least one shared metric and worsens none.

    `attempts` and other costs are lower-is-better. Success rates are higher-is-better.
    """
    shared = set(baseline) & set(candidate)
    if not shared:
        return False
    improved = False
    success_b = float(baseline.get("learning_success_rate", 0) or 0)
    success_c = float(candidate.get("learning_success_rate", 0) or 0)
    success_up = success_c > success_b
    for k in shared:
        if k == "known_skill_ok":
            continue
        b, c = float(baseline[k]), float(candidate[k])
        cost = k in _LOWER_IS_BETTER or k.endswith("_cost") or k.endswith("_time")
        if cost:
            if success_up:
                if c < b:
                    improved = True
                continue
            if c > b:
                return False
            if c < b:
                improved = True
        else:
            if c < b:
                return False
            if c > b:
                improved = True
    return improved


def _violates_invariants(snap: LearnerSnapshot) -> list[str]:
    have = set(snap.invariants or [])
    return sorted(PROTECTED_INVARIANTS - have)


def benchmark_learner(snap: LearnerSnapshot, *, succeed_on_attempt: int = 2) -> dict[str, float]:
    """Run the shipped learner configured by this snapshot. Metrics are measured, not planted."""
    from .learning import LearnContext, learn_goal
    from .memory_kinds import TypedMemory
    from .resolver import assess_capability
    from .self_model import snapshot_self_model
    from .skill_store import SkillStore

    from .records import InterpretedIntent

    box = {"n": 0}

    def observer() -> dict[str, Any]:
        ok = box["n"] >= succeed_on_attempt
        return {"pose": "ready" if ok else "home", "goal_met": ok, "trajectory_ok": ok and snap.trajectory_weight <= 0}

    def executor(_plan: list[dict[str, Any]]) -> dict[str, Any]:
        box["n"] += 1
        return {"ok": True, "ran": True}

    def interp(req, skills, model):
        return InterpretedIntent(
            goal="composed_outcome",
            required_capabilities=["balance", "weight_shift"],
            domain="embodied",
            expected_outcome=req,
            confidence=0.5,
        )

    ctx = LearnContext(
        skills=SkillStore(seed=True),
        memory=TypedMemory(),
        self_model=snapshot_self_model(),
        max_attempts=int(snap.max_attempts),
        executor=executor,
        observer=observer,
        snapshot=snap,
        interpreter=interp,
    )
    result = learn_goal("novel locomotion sequence", ctx)
    catalog = SkillStore(seed=True)
    known = next(s for s in catalog.all() if s.kind != "primitive" and s.validated and s.supported_goals)
    greet = assess_capability(
        known.supported_goals[-1],
        skills=catalog.only([known.skill_id]),
        retrieval_threshold=float(snap.retrieval_threshold),
    )
    return {
        "learning_success_rate": 1.0 if result.success else 0.0,
        "attempts": float(result.attempts),
        "known_skill_ok": 1.0 if greet.decision == "execute" else 0.0,
    }


class RSIPipeline:
    def __init__(self, root: Path | None = None, events: EventLog | None = None) -> None:
        self.root = Path(root) if root is not None else None
        self.events = events or EventLog()
        self.active = default_snapshot()
        self.previous: LearnerSnapshot | None = None
        self.records: list[ImprovementRecord] = []
        self._load()

    def _dir(self) -> Path | None:
        if self.root is None:
            return None
        p = self.root / "rsi"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _load(self) -> None:
        d = self._dir()
        if d is None:
            return
        active = d / "active.json"
        prev = d / "previous.json"
        journal = d / "journal.jsonl"
        if active.is_file():
            try:
                self.active = LearnerSnapshot.from_dict(json.loads(active.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        if prev.is_file():
            try:
                self.previous = LearnerSnapshot.from_dict(json.loads(prev.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError):
                self.previous = None
        if journal.is_file():
            for line in journal.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.records.append(ImprovementRecord(**{k: v for k, v in row.items() if k in ImprovementRecord.__dataclass_fields__}))

    def _save_active(self) -> None:
        d = self._dir()
        if d is None:
            return
        (d / "active.json").write_text(json.dumps(self.active.to_dict(), indent=2), encoding="utf-8")
        if self.previous is not None:
            (d / "previous.json").write_text(json.dumps(self.previous.to_dict(), indent=2), encoding="utf-8")

    def _journal(self, rec: ImprovementRecord) -> None:
        self.records.append(rec)
        d = self._dir()
        if d is None:
            return
        with (d / "journal.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict()) + "\n")

    def observe_weakness(
        self,
        *,
        subsystem: str,
        problem: str,
        evidence: list[str],
        frequency: int,
        hypothesis: str,
        proposed_improvement: str,
        expected_benefit: str,
        risk: str,
        measurable_success_criteria: str,
    ) -> ImprovementObservation:
        obs = ImprovementObservation(
            subsystem=subsystem,
            problem=problem,
            evidence=list(evidence),
            frequency=int(frequency),
            hypothesis=hypothesis,
            proposed_improvement=proposed_improvement,
            expected_benefit=expected_benefit,
            risk=risk,
            measurable_success_criteria=measurable_success_criteria,
        )
        self.events.emit("improvement_observed", subsystem=subsystem, frequency=frequency)
        self.events.emit("improvement_proposed", hypothesis=hypothesis)
        return obs

    def build_candidate(
        self,
        obs: ImprovementObservation,
        change: dict[str, Any],
    ) -> tuple[LearnerSnapshot, ImprovementRecord]:
        cand = copy.deepcopy(self.active)
        cand.version = f"{self.active.version}+{uuid4().hex[:6]}"
        for key in ("retrieval_threshold", "max_attempts", "trajectory_weight", "replay_penalty"):
            if key in change:
                setattr(cand, key, change[key])
        if "invariants" in change and isinstance(change["invariants"], list):
            cand.invariants = list(change["invariants"])
        rec = ImprovementRecord(
            improvement_id=uuid4().hex[:12],
            timestamp=_now(),
            originating_problem=obs.problem,
            evidence=list(obs.evidence),
            hypothesis=obs.hypothesis,
            affected_subsystem=obs.subsystem,
            baseline_version=self.active.version,
            candidate_version=cand.version,
            proposed_change=dict(change),
            baseline_metrics={},
            candidate_metrics={},
            regression_results={},
            safety_results={},
            decision=CandidateState.BUILDING.value,
            promoted=False,
            rollback_target=self.active.version,
            state=CandidateState.BUILDING.value,
            observation=asdict(obs),
        )
        self.events.emit("candidate_built", candidate_version=cand.version)
        rec.state = CandidateState.PROPOSED.value
        rec.decision = CandidateState.PROPOSED.value
        self._journal(rec)
        return cand, rec

    def evaluate_candidate(
        self,
        cand: LearnerSnapshot,
        rec: ImprovementRecord,
        *,
        baseline_metrics: dict[str, float] | None = None,
        candidate_metrics: dict[str, float] | None = None,
        existing_capability_ok: bool | None = None,
        scorer: Callable[[dict[str, float], dict[str, float]], bool] | None = None,
        benchmark: Callable[[LearnerSnapshot], dict[str, float]] | None = None,
    ) -> ImprovementRecord:
        run = benchmark or benchmark_learner
        if baseline_metrics is None:
            baseline_metrics = run(self.active)
        if candidate_metrics is None:
            candidate_metrics = run(cand)
        rec.baseline_metrics = dict(baseline_metrics)
        rec.candidate_metrics = dict(candidate_metrics)
        rec.state = CandidateState.TESTING.value
        rec.decision = CandidateState.TESTING.value
        dropped = _violates_invariants(cand)
        rec.safety_results = {"invariant_violations": dropped}
        self.events.emit("candidate_evaluated", candidate_version=cand.version, violations=dropped)
        if dropped:
            rec.state = CandidateState.FAILED.value
            rec.decision = CandidateState.FAILED.value
            rec.promoted = False
            rec.regression_results = {"invariants": dropped}
            self.events.emit("candidate_rejected", reason="invariant_violation", dropped=dropped)
            self._journal(rec)
            return rec
        if existing_capability_ok is None:
            existing_capability_ok = float(candidate_metrics.get("known_skill_ok", 1.0)) >= 1.0
        if not existing_capability_ok:
            rec.state = CandidateState.REGRESSED.value
            rec.decision = CandidateState.REGRESSED.value
            rec.promoted = False
            rec.regression_results = {"existing_capability_ok": False}
            self.events.emit("candidate_rejected", reason="regression")
            self._journal(rec)
            return rec
        better = True
        if scorer is not None:
            better = bool(scorer(baseline_metrics, candidate_metrics))
        else:
            better = metrics_candidate_better(baseline_metrics, candidate_metrics)
        if not better:
            rec.state = CandidateState.FAILED.value
            rec.decision = CandidateState.FAILED.value
            rec.promoted = False
            rec.regression_results = {"not_better": True}
            self.events.emit("candidate_rejected", reason="not_better")
            self._journal(rec)
            return rec
        rec.state = CandidateState.PASSED.value
        rec.decision = CandidateState.AWAITING_PROMOTION.value
        rec.promoted = False
        self.events.emit("candidate_passed", candidate_version=cand.version)
        self._journal(rec)
        return rec

    def promote(self, cand: LearnerSnapshot, rec: ImprovementRecord) -> ImprovementRecord:
        if rec.state not in {CandidateState.PASSED.value, CandidateState.AWAITING_PROMOTION.value} and rec.decision != CandidateState.AWAITING_PROMOTION.value:
            rec.state = CandidateState.FAILED.value
            rec.decision = CandidateState.FAILED.value
            rec.promoted = False
            self.events.emit("candidate_rejected", reason="not_promotion_ready")
            self._journal(rec)
            return rec
        if _violates_invariants(cand):
            rec.state = CandidateState.FAILED.value
            rec.decision = CandidateState.FAILED.value
            rec.promoted = False
            self.events.emit("candidate_rejected", reason="invariant_on_promote")
            self._journal(rec)
            return rec
        self.previous = copy.deepcopy(self.active)
        self.active = copy.deepcopy(cand)
        rec.state = CandidateState.ACTIVE.value
        rec.decision = CandidateState.ACTIVE.value
        rec.promoted = True
        rec.rollback_target = self.previous.version
        self._save_active()
        self.events.emit("candidate_promoted", candidate_version=cand.version, rollback=self.previous.version)
        self._journal(rec)
        return rec

    def rollback(self, rec: ImprovementRecord | None = None, *, post_metrics: dict[str, float] | None = None) -> ImprovementRecord:
        if self.previous is None:
            raise RuntimeError("no rollback target")
        target = self.previous.version
        self.active, self.previous = copy.deepcopy(self.previous), copy.deepcopy(self.active)
        self._save_active()
        row = rec or ImprovementRecord(
            improvement_id=uuid4().hex[:12],
            timestamp=_now(),
            originating_problem="post-deployment regression",
            evidence=[],
            hypothesis="rollback",
            affected_subsystem="learner",
            baseline_version=self.active.version,
            candidate_version=target,
            proposed_change={},
            baseline_metrics={},
            candidate_metrics=dict(post_metrics or {}),
            regression_results={"post_deployment": True},
            safety_results={},
            decision=CandidateState.ROLLED_BACK.value,
            promoted=False,
            rollback_target=self.active.version,
            state=CandidateState.ROLLED_BACK.value,
        )
        row.state = CandidateState.ROLLED_BACK.value
        row.decision = CandidateState.ROLLED_BACK.value
        row.promoted = False
        row.rollback_target = self.active.version
        row.post_deployment_metrics = dict(post_metrics or {})
        self.events.emit("rollback_performed", restored=self.active.version)
        self._journal(row)
        return row

    def active_copy(self) -> LearnerSnapshot:
        return copy.deepcopy(self.active)
