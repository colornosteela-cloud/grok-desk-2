"""Capability resolver: facts + skills + interpreted intent → decision.

This module must not mention user action names. Matching uses interpreter
output and skill/self-model records only.
"""

from __future__ import annotations

from typing import Any, Iterable

from .events import EventLog
from .interpreter import GoalInterpreter, phrase_score, record_backed_interpreter
from .records import CapabilityAssessment, CompiledSkill, InterpretedIntent
from .self_model import SelfModel, snapshot_self_model
from .skill_store import SkillStore


def _ids(skills: Iterable[CompiledSkill]) -> set[str]:
    out: set[str] = set()
    for sk in skills:
        out.add(sk.skill_id)
        out.add(sk.skill_id.rsplit(".", 1)[-1])
        out.add(sk.name)
        for p in sk.required_capabilities:
            out.add(p)
        if sk.kind == "primitive":
            out.add(sk.skill_id.rsplit(".", 1)[-1])
    return out


def _semantic_skill_score(intent: InterpretedIntent, sk: CompiledSkill) -> float:
    """Map a free-text Qwen goal onto a skill via name/description, not planted paraphrases."""
    query = " ".join(
        [intent.goal or "", intent.expected_outcome or ""] + list(intent.required_capabilities or [])
    )
    score = 0.0
    for blob in (sk.name, sk.semantic_description, sk.expected_outcome or ""):
        if not blob:
            continue
        score = max(score, phrase_score(query, blob), phrase_score(blob, query))
    return score


def _match_skill(intent: InterpretedIntent, skills: list[CompiledSkill]) -> CompiledSkill | None:
    # Descriptions of an action are not permission to execute that action.
    if intent.domain in {"talk", "perception"}:
        return None
    wanted = set(intent.required_capabilities)
    hits: list[CompiledSkill] = []
    if intent.goal and not intent.goal.startswith("unknown") and not intent.goal.startswith("composed"):
        root = intent.goal.split(".v")[0]
        for sk in skills:
            if sk.kind == "primitive":
                continue
            base = sk.skill_id.split(".v")[0]
            if sk.skill_id == intent.goal or sk.name == intent.goal or base == root:
                hits.append(sk)
    if not hits:
        for sk in skills:
            if sk.kind == "primitive" or not sk.validated:
                continue
            need = set(sk.required_capabilities or [sk.skill_id])
            # A skill whose primitives are a subset of the request is compose
            # material, not an EXECUTE hit — otherwise "raise arm" matches point.
            if sk.skill_id in wanted:
                hits.append(sk)
            elif need and need <= wanted and intent.confidence >= 0.85:
                if _semantic_skill_score(intent, sk) >= 0.5:
                    hits.append(sk)
    if not hits:
        best: CompiledSkill | None = None
        best_s = 0.0
        for sk in skills:
            if sk.kind == "primitive" or not sk.validated:
                continue
            s = _semantic_skill_score(intent, sk)
            if s > best_s:
                best, best_s = sk, s
        if best is not None and best_s >= 0.5:
            hits.append(best)
    if not hits:
        return None
    validated = [s for s in hits if s.validated]
    pool = validated or hits
    return max(pool, key=lambda s: (int(s.version or 1), s.skill_id))


def assess_capability(
    request: str,
    state: Any = None,
    *,
    skills: SkillStore | list[CompiledSkill] | None = None,
    self_model: SelfModel | None = None,
    interpreter: GoalInterpreter | None = None,
    events: EventLog | None = None,
    retrieval_threshold: float = 0.85,
) -> CapabilityAssessment:
    """Decide EXECUTE / COMPOSE / LEARN / RESEARCH / OBSERVE / PRACTICE / ASK / REFUSE."""
    store: SkillStore | None = None
    skill_list: list[CompiledSkill]
    if isinstance(state, SkillStore):
        store = state
        skill_list = state.all()
    elif hasattr(state, "skills") and state is not None and not isinstance(state, (str, bytes)):
        raw = getattr(state, "skills")
        if isinstance(raw, SkillStore):
            store = raw
            skill_list = raw.all()
        elif isinstance(raw, list):
            # compatibility: old TeelaState with capability.Skill objects
            skill_list = []
            for item in raw:
                if isinstance(item, CompiledSkill):
                    skill_list.append(item)
                else:
                    skill_list.append(
                        CompiledSkill(
                            skill_id=str(getattr(item, "skill_id", "")),
                            name=str(getattr(item, "skill_id", "")).rsplit(".", 1)[-1],
                            semantic_description=str(getattr(item, "description", "")),
                            supported_goals=list(getattr(item, "goals", []) or []),
                            required_capabilities=list(getattr(item, "primitives", []) or []),
                            kind=str(getattr(item, "kind", "skill")),
                            validated=bool(getattr(item, "validated", False)),
                            confidence=float(getattr(item, "confidence", 0.5) or 0.5),
                            tool=getattr(item, "tool", None),
                            tool_args=getattr(item, "tool_args", None),
                        )
                    )
        else:
            skill_list = []
    elif isinstance(skills, SkillStore):
        store = skills
        skill_list = skills.all()
    elif isinstance(skills, list):
        skill_list = skills
    else:
        store = SkillStore(seed=True)
        skill_list = store.all()

    model = self_model
    if model is None and state is not None and getattr(state, "self_model", None) is not None:
        model = state.self_model
    if model is None:
        model = snapshot_self_model(
            learned_skill_ids=[s.skill_id for s in skill_list if s.kind != "primitive"],
            tools=[s.tool for s in skill_list if s.tool],
        )
    threshold = float(retrieval_threshold or 0.85)
    if interpreter is not None:
        interpreted = interpreter(request, skill_list, model)
    else:
        interpreted = record_backed_interpreter(
            request, skill_list, model, threshold=threshold
        )
    log = events or EventLog()
    log.emit(
        "goal_interpreted",
        goal=interpreted.goal,
        domain=interpreted.domain,
        required=interpreted.required_capabilities,
    )

    available = set(model.available_primitive_ids()) | _ids(skill_list)
    for cap in model.available_tool_ids():
        available.add(cap)
        available.add(cap.rsplit(".", 1)[-1])
    for uid in model.unavailable:
        available.discard(uid)

    required = list(interpreted.required_capabilities)
    missing = [c for c in required if c not in available and c.rsplit(".", 1)[-1] not in available]
    matched = _match_skill(interpreted, skill_list)
    matched_ids = [matched.skill_id] if matched is not None else []

    decision = "learn"
    reason = "No validated way to satisfy the goal."
    executable = False
    composable = False
    learnable = True
    conf = float(interpreted.confidence)

    if interpreted.domain == "talk":
        decision, executable, learnable, reason, conf = (
            "execute",
            True,
            False,
            "No new skill is required for this turn.",
            1.0,
        )
    elif matched is not None and matched.validated and interpreted.confidence >= threshold:
        decision, executable, learnable, reason = (
            "execute",
            True,
            False,
            "Existing validated skill satisfies requested outcome.",
        )
        conf = max(conf, float(matched.confidence))
        log.emit("skill_resolved", skill_id=matched.skill_id)
    elif interpreted.goal == "unclear":
        decision, learnable, reason = (
            "ask",
            True,
            "The requested outcome is not clear enough to execute or compose.",
        )
    elif interpreted.domain == "software" and not required:
        decision, learnable, reason = (
            "research" if model.permissions.get("network", True) else "ask",
            True,
            "Software knowledge is missing; research or ask before acting.",
        )
    elif interpreted.domain == "software" and not missing:
        decision, executable, learnable, reason = (
            "execute",
            True,
            False,
            "Existing computer capabilities can satisfy the requested outcome.",
        )
    elif interpreted.domain == "software" and missing:
        decision, reason = (
            "research" if model.permissions.get("network", True) else "ask",
            "Software knowledge is missing.",
        )
        learnable = True
    elif interpreted.domain == "perception":
        decision, executable, learnable, reason = (
            "observe",
            True,
            False,
            "The requested outcome is a perception check, not a new skill.",
        )
    elif not required:
        # Specific unmatched goal: learn. ASK is reserved for goal == "unclear".
        decision, learnable, reason = (
            "learn",
            True,
            "Teela does not currently possess a validated way to satisfy the goal.",
        )
    elif not missing and required:
        practice = float(interpreted.confidence) < 0.45
        decision, composable, learnable, reason = (
            "practice" if practice else "compose",
            True,
            True,
            "No direct skill exists, but existing primitives can satisfy the goal.",
        )
        log.emit("skill_composition_started", required=required)
    elif missing and not model.permissions.get("motors", True) and interpreted.domain == "embodied":
        # motors permission false still allows virtual twin; refuse only if primitive absent
        decision, learnable, reason = (
            "learn",
            True,
            "Teela does not currently possess a validated way to satisfy the goal.",
        )
    elif missing:
        # If the missing items are hardware the self-model says is unavailable, refuse.
        hard_miss = [m for m in missing if m in model.unavailable]
        if hard_miss and not any(model.has(m) for m in missing):
            decision, learnable, reason = (
                "refuse",
                False,
                "Required capability is unavailable on this body and cannot be learned here.",
            )
        else:
            decision, learnable, reason = (
                "learn",
                True,
                "Teela does not currently possess a validated way to satisfy the goal.",
            )
    else:
        decision, learnable, reason = (
            "learn",
            True,
            "Teela does not currently possess a validated way to satisfy the goal.",
        )

    assessment = CapabilityAssessment(
        goal=interpreted.goal,
        interpreted_intent=request,
        required_capabilities=required,
        matched_skills=matched_ids,
        available_primitives=sorted(model.available_primitive_ids()),
        missing_capabilities=missing,
        missing_knowledge=(
            [f"procedure for {interpreted.expected_outcome}"]
            if missing or decision in {"learn", "research", "practice"}
            else []
        ),
        confidence=conf,
        executable=executable,
        composable=composable,
        learnable=learnable,
        decision=decision,
        reason=reason,
        intent=request,
        capability=matched_ids[0] if matched_ids else None,
        known=bool(matched_ids) or interpreted.domain in {"talk", "software"} and executable,
        missing_primitives=missing,
        domain=interpreted.domain,
    )
    log.emit("capability_assessed", decision=decision, reason=reason, matched=matched_ids)
    return assessment


def needs_learn_attempt(request: str, state: Any = None, **kwargs: Any) -> bool:
    """True when compose/learn/research/practice. No skill names here."""
    a = assess_capability(request, state, **kwargs)
    return a.decision in {"compose", "learn", "research", "practice"}
