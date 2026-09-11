"""Domain-agnostic learning loop. No per-skill special cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

from .events import EventLog
from .memory_kinds import TypedMemory
from .outcome import evaluate_outcome
from .records import (
    CompiledSkill,
    ExperienceRecord,
    LearningResult,
    OutcomeEvaluation,
    _now,
)
from .resolver import assess_capability
from .records import LearnerSnapshot
from .self_model import SelfModel
from .skill_store import SkillStore


def compose_plan(
    goal: str,
    required: list[str],
    *,
    prior_plan: list[dict[str, Any]] | None = None,
    discrepancy: str | None = None,
    replay_penalty: float = 1.0,
) -> list[dict[str, Any]]:
    """Build a candidate plan from required capabilities. Revises if a prior plan failed."""
    cap = list(required[:6])
    if not cap:
        # No capability evidence is not authorization for a locomotion trial.
        return []
    steps = [{"capability": c, "action": "apply", "goal": goal} for c in cap]
    if prior_plan:
        rest = list(prior_plan[1:]) if len(prior_plan) > 1 else []
        extra = {
            "capability": cap[-1],
            "action": "adjust",
            "because": discrepancy or "prior failure",
            "penalty": float(replay_penalty),
        }
        if float(replay_penalty) > 1.5:
            steps = list(reversed(rest)) + [extra, {"capability": cap[0], "action": "retry_alt", "goal": goal}]
        else:
            steps = rest + [extra, {"capability": "stand", "action": "stabilize", "goal": goal}]
        if steps == prior_plan:
            steps = list(prior_plan) + [extra]
    return steps


def _simulate(plan: list[dict[str, Any]], self_model: SelfModel) -> dict[str, Any]:
    unsafe = []
    for step in plan:
        cap = str(step.get("capability") or "")
        if cap in self_model.unavailable:
            unsafe.append(cap)
    return {"safe": not unsafe, "unsafe": unsafe, "plan": plan}


@dataclass
class LearnContext:
    skills: SkillStore
    memory: TypedMemory
    self_model: SelfModel
    events: EventLog = field(default_factory=EventLog)
    max_attempts: int = 4
    executor: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None
    observer: Callable[[], dict[str, Any]] | None = None
    assertions: list | None = None
    acquire_knowledge: Callable[[str], dict[str, Any]] | None = None
    dry_run: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None
    snapshot: LearnerSnapshot | None = None
    interpreter: Any = None


def learn_goal(goal: str, ctx: LearnContext, *, request: str | None = None) -> LearningResult:
    """Inspect → compose/research → trial → evaluate → persist or revise."""
    text = request or goal
    snap = ctx.snapshot or LearnerSnapshot(version="learner_v1")
    ctx.max_attempts = int(snap.max_attempts)
    ctx.events.emit("learning_started", goal=goal)
    assessment = assess_capability(
        text,
        skills=ctx.skills,
        self_model=ctx.self_model,
        events=ctx.events,
        retrieval_threshold=float(snap.retrieval_threshold),
        interpreter=ctx.interpreter,
    )
    related = ctx.memory.related(goal)
    plan: list[dict[str, Any]] = []
    last_eval: OutcomeEvaluation | None = None
    last_exec: dict[str, Any] = {}
    # Conversation, observation, and already-executable work have no trial.
    # Software *learning* may still run; locomotion is not a stand-in for it.
    _domain = str(getattr(assessment, "domain", "") or "")
    if assessment.decision in {"observe", "execute"} or _domain in {"talk", "perception"}:
        return LearningResult(
            False, assessment.decision, 0, None, None, [],
            "No learning trial required; dispatch the resolved outcome normally.",
            ctx.events.kinds(),
        )
    if assessment.decision == "refuse":
        return LearningResult(False, "refuse", 0, None, None, [], assessment.reason, ctx.events.kinds())
    if assessment.decision == "ask":
        return LearningResult(False, "ask", 0, None, None, [], assessment.reason, ctx.events.kinds())

    if (
        assessment.missing_knowledge or assessment.decision in {"learn", "research", "practice"}
    ) and ctx.acquire_knowledge is not None:
        info = ctx.acquire_knowledge(goal)
        ctx.memory.add_knowledge(str(info.get("text") or info), tags=["learning", goal])

    required = list(assessment.required_capabilities)
    if not required and assessment.decision in {"learn", "practice", "compose", "research"}:
        if assessment.decision == "research" or _domain == "software":
            required = ["software.research"]
        elif _domain == "embodied":
            required = ["stand", "step"]
        else:
            required = []
    prior: list[dict[str, Any]] | None = None
    discrepancy: str | None = None
    if related:
        prior = list(related[-1].plan)
        discrepancy = (related[-1].discrepancies or ["prior failure"])[0] if not related[-1].success else None

    for attempt in range(1, int(ctx.max_attempts) + 1):
        plan = compose_plan(
            goal,
            required,
            prior_plan=prior,
            discrepancy=discrepancy,
            replay_penalty=float(snap.replay_penalty),
        )
        ctx.events.emit("learning_attempt", attempt=attempt, plan=plan)
        sim = (ctx.dry_run or (lambda p: _simulate(p, ctx.self_model)))(plan)
        ctx.events.emit("simulation_result", safe=bool(sim.get("safe")), unsafe=sim.get("unsafe"))
        if not sim.get("safe", True):
            prior, discrepancy = plan, "unsafe simulation"
            continue
        before = ctx.observer() if ctx.observer else {}
        if ctx.executor is not None:
            last_exec = ctx.executor(plan) or {}
        else:
            last_exec = {"ok": True, "ran": True}
        ctx.events.emit("execution_result", ok=bool(last_exec.get("ok")), attempt=attempt)
        after = ctx.observer() if ctx.observer else dict(before)
        last_eval = evaluate_outcome(
            expected=assessment.interpreted_intent or goal,
            before=before,
            after=after,
            execution=last_exec,
            assertions=ctx.assertions,
            events=ctx.events,
            trajectory_weight=float(snap.trajectory_weight),
        )
        ctx.memory.add_experience(
            ExperienceRecord(
                experience_id=uuid4().hex[:12],
                goal=goal,
                plan=plan,
                success=last_eval.success,
                evidence=last_eval.evidence,
                discrepancies=last_eval.discrepancies,
            )
        )
        if last_eval.success:
            skill = CompiledSkill(
                skill_id=f"learned.{goal.replace(' ', '_')[:40]}.{uuid4().hex[:6]}",
                name=goal[:48],
                semantic_description=f"Learned procedure for: {goal}",
                supported_goals=[goal, text],
                required_capabilities=required,
                plan=plan,
                expected_outcome=goal,
                confidence=last_eval.confidence,
                successes=1,
                source="learned",
                validated=True,
                last_validated=_now(),
                kind="skill",
            )
            ctx.events.emit("skill_compiled", skill_id=skill.skill_id)
            ctx.skills.put(skill)
            ctx.events.emit("skill_validated", skill_id=skill.skill_id)
            ctx.events.emit("skill_persisted", skill_id=skill.skill_id)
            return LearningResult(True, "execute", attempt, skill, last_eval, plan, "learned and stored", ctx.events.kinds())
        prior = plan
        discrepancy = (last_eval.discrepancies or ["outcome not met"])[0]
    return LearningResult(
        False,
        assessment.decision,
        ctx.max_attempts,
        None,
        last_eval,
        plan,
        "Could not achieve the goal within learning budget.",
        ctx.events.kinds(),
    )
