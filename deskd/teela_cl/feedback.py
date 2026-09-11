"""Conversational correction loop. Exemplars live in data; routing has no phrase gates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .attempts import AttemptStore
from .events import EventLog
from .interpreter import phrase_score
from .learning import compose_plan, _simulate
from .outcome import evaluate_outcome
from .records import (
    AttemptRecord,
    CompiledSkill,
    ErrorClass,
    FeedbackType,
    UserFeedback,
    _now,
)
from .rsi import RSIPipeline
from .self_model import SelfModel
from .skill_store import SkillStore


SYSTEMIC_THRESHOLD = 3


def feedback_exemplars() -> list[dict[str, Any]]:
    """Paraphrases of feedback *kinds*. Not consulted by diagnose/error-class routing."""
    return [
        {
            "feedback_type": FeedbackType.POSITIVE.value,
            "supported": [
                "yes that's right",
                "that's right",
                "that's exactly it",
                "yes exactly",
                "much better",
                "that's better",
                "remember how you just did that",
                "perfect",
                "that's the one",
                "keep doing it that way",
            ],
        },
        {
            "feedback_type": FeedbackType.NEGATIVE.value,
            "supported": [
                "that wasn't right",
                "that's not right",
                "that's wrong",
                "no that's not right",
                "nah",
                "nope",
                "no",
                "that looked awkward",
                "you didn't do that right",
                "not quite",
                "that's not quite what i meant",
            ],
        },
        {
            "feedback_type": FeedbackType.CORRECTION.value,
            "supported": [
                "keep the upper arm still",
                "motion from the wrist",
                "arm needs to be higher",
                "move more slowly",
                "don't turn the whole body",
                "try again but use the other hand",
                "the other hand",
                "hold the shoulder",
                "bend only the wrist",
                "don't move your whole arm",
                "wave from your wrist",
                "use your wrist",
                "not your whole arm",
            ],
        },
        {
            "feedback_type": FeedbackType.PREFERENCE.value,
            "supported": [
                "i prefer it slower",
                "a little smaller",
                "more gently",
            ],
        },
        {
            "feedback_type": FeedbackType.CLARIFICATION.value,
            "supported": [
                "i meant the other one",
                "not that object the other",
            ],
        },
    ]


class FeedbackInterpreter(Protocol):
    def __call__(self, message: str, attempt: AttemptRecord | None, self_model: SelfModel) -> UserFeedback: ...


def _feedback_score(message: str, goal: str) -> float:
    """Match whole feedback phrases. A single shared word is not enough."""
    from .interpreter import _norm, _tokens

    r, g = _norm(message), _norm(goal)
    if not g:
        return 0.0
    if g == r:
        return 1.0
    if re.search(rf"(?:^| ){re.escape(g)}(?:$| )", r):
        return 1.0
    if len(_tokens(g)) < 2:
        return 0.0
    return phrase_score(message, goal)


def _best_exemplar(message: str) -> tuple[str, float]:
    best_type, best = FeedbackType.NONE.value, 0.0
    for row in feedback_exemplars():
        local = 0.0
        for g in row.get("supported") or []:
            local = max(local, _feedback_score(message, str(g)))
        if local > best:
            best_type, best = str(row["feedback_type"]), local
    if best < 0.85:
        return FeedbackType.NONE.value, 0.0
    return best_type, best


def _is_social_talk(message: str) -> bool:
    from .interpreter import _norm, _talk

    if _talk(message):
        return True
    t = _norm(message)
    return t.startswith(("hi ", "hello ", "hey ", "thanks", "thank you", "good morning", "good night"))


def _is_short_followup(message: str) -> bool:
    """Relative adjustment in an active attempt — not greetings, not new tasks."""
    t = " ".join((message or "").lower().split())
    if not t or _is_social_talk(t):
        return False
    if t.endswith("?") and not t.startswith(("no", "was", "is that")):
        return False
    words = [w.strip(".,!") for w in t.split() if w.strip(".,!")]
    if not (1 <= len(words) <= 4):
        return False
    try:
        import robot_sim

        if robot_sim.is_delta_followup(t):
            return True
    except Exception:
        pass
    kind, score = _best_exemplar(t)
    if kind != FeedbackType.NONE.value and score >= 0.85:
        return True
    # Comparative / retry fragments in an open attempt (not skill names).
    blob = " ".join(words)
    return any(
        p in blob
        for p in ("more", "less", "again", "higher", "lower", "slower", "faster", "no", "nah", "nope")
    )


def _is_manner_correction(message: str) -> bool:
    """True when the utterance modifies how the last attempt should be done."""
    t = (message or "").lower()
    return bool(
        re.search(
            r"\b(?:don't|do not|from your|instead|only the|keep the|not your whole|"
            r"the other (?:hand|arm|side)|use your)\b",
            t,
        )
    )


def _is_new_action_request(message: str) -> bool:
    """A new outcome to perform is not commentary on the last attempt."""
    t = " ".join((message or "").lower().split())
    if not t or _is_social_talk(t):
        return False
    if _is_manner_correction(t):
        return False
    kind, score = _best_exemplar(t)
    if kind != FeedbackType.NONE.value and score >= 0.85:
        return False
    if t.startswith("can you") or t.startswith("could you") or t.startswith("please "):
        return True
    try:
        import robot_sim

        if robot_sim.looks_like_motor(t) or robot_sim.infer_command(t):
            return True
    except Exception:
        pass
    try:
        import virtual_body

        if virtual_body.teela_args_from_intent(t):
            return True
    except Exception:
        pass
    return False


def _primitives_from_text(message: str, self_model: SelfModel) -> tuple[list[str], list[str]]:
    """Map mentioned body parts / primitive descriptions onto hold vs actuate."""
    text = (message or "").lower()
    desired: list[str] = []
    holds: list[str] = []
    hold_hint = any(
        phrase_score(text, cue) >= 0.4
        for cue in ("keep still", "hold still", "mostly still", "stationary", "don't move")
    )
    for fact in self_model.primitives:
        if not fact.available:
            continue
        score = max(
            phrase_score(text, fact.description),
            phrase_score(text, fact.cap_id.replace("_", " ")),
        )
        tail = fact.cap_id.rsplit(".", 1)[-1]
        part = tail.replace("hold_", "").replace("oscillate_", "").replace("bend_", "").replace("raise_", "")
        if part and part in text:
            score = max(score, 0.6)
        if "upper arm" in text and tail in {"hold_shoulder", "raise_arm"}:
            score = max(score, 0.7)
        if score < 0.45:
            continue
        if hold_hint and fact.family == "arm" and not tail.startswith("oscillate"):
            hid = tail if tail.startswith("hold_") else f"hold_{part}" if part in {"shoulder", "elbow"} else tail
            if hid not in holds:
                holds.append(hid)
        elif tail not in desired:
            desired.append(tail)
    return desired, holds


def interpret_feedback(
    message: str,
    attempt: AttemptRecord | None,
    self_model: SelfModel,
    *,
    interpreter: FeedbackInterpreter | None = None,
) -> UserFeedback:
    if interpreter is not None:
        return interpreter(message, attempt, self_model)
    kind, score = _best_exemplar(message)
    if (
        kind == FeedbackType.NONE.value
        and attempt is not None
        and re.search(
            r"\b(?:try it again|do (?:it|that) again|once more|didn'?t see you)\b",
            (message or "").lower(),
        )
        and not _is_social_talk(message)
    ):
        kind, score = FeedbackType.NEGATIVE.value, 0.8
    if kind == FeedbackType.NONE.value and _is_new_action_request(message):
        return UserFeedback(
            feedback_type=FeedbackType.NONE.value,
            target_goal_id="",
            target_attempt_id=None,
            user_message=message,
            confidence=0.0,
        )
    if kind == FeedbackType.NONE.value and attempt is not None and _is_short_followup(message):
        kind, score = FeedbackType.CORRECTION.value, 0.7
    desired, holds = _primitives_from_text(message, self_model)
    if kind == FeedbackType.NEGATIVE.value and (desired or holds):
        kind = FeedbackType.CORRECTION.value
        score = max(score, 0.7)
    if kind == FeedbackType.NONE.value and (desired or holds) and attempt is not None:
        kind, score = FeedbackType.CORRECTION.value, 0.55
    if kind == FeedbackType.NONE.value:
        return UserFeedback(
            feedback_type=kind,
            target_goal_id="",
            target_attempt_id=None,
            user_message=message,
            confidence=0.0,
        )
    problem = None
    change = None
    if holds:
        problem = "too much proximal motion" if holds else None
        change = "hold " + ", ".join(holds)
    if desired:
        change = ((change + "; ") if change else "") + "use " + ", ".join(desired)
    return UserFeedback(
        feedback_type=kind,
        target_goal_id=(attempt.goal_id if attempt else ""),
        target_attempt_id=(attempt.attempt_id if attempt else None),
        user_message=message,
        inferred_problem=problem,
        desired_change=change,
        constraints=holds,
        desired_primitives=desired,
        confidence=float(score),
    )


def diagnose(
    feedback: UserFeedback,
    attempt: AttemptRecord | None,
    store: AttemptStore,
) -> str:
    """Error class from structured feedback + attempt evidence. No user phrases here."""
    ft = feedback.feedback_type
    if ft == FeedbackType.POSITIVE.value:
        return ErrorClass.VALIDATION.value
    if attempt is None:
        return ErrorClass.ASK.value
    if ft == FeedbackType.CLARIFICATION.value:
        return ErrorClass.MISUNDERSTANDING.value
    if feedback.constraints or feedback.desired_primitives or feedback.desired_change:
        n = store.negatives_for(skill=attempt.skill) if attempt.skill else store.negatives_for(goal=attempt.goal)
        if n + 1 >= SYSTEMIC_THRESHOLD:
            return ErrorClass.SYSTEM.value
        return ErrorClass.SKILL.value
    unexpected = (attempt.evaluation or {}).get("unexpected_effects") or []
    if unexpected:
        return ErrorClass.EXECUTION.value
    if feedback.confidence < 0.35:
        return ErrorClass.ASK.value
    if ft == FeedbackType.PREFERENCE.value:
        return ErrorClass.PLANNING.value
    n = store.negatives_for(goal_id=attempt.goal_id) if attempt.goal_id else 0
    if n + 1 >= SYSTEMIC_THRESHOLD:
        return ErrorClass.SYSTEM.value
    if ft == FeedbackType.NEGATIVE.value:
        return ErrorClass.EXECUTION.value
    return ErrorClass.SKILL.value


def propose_corrected_plan(
    attempt: AttemptRecord,
    feedback: UserFeedback,
    *,
    error_class: str,
) -> list[dict[str, Any]]:
    prior = list(attempt.plan or [])
    if error_class == ErrorClass.EXECUTION.value:
        return prior or compose_plan(attempt.goal, [])
    caps: list[str] = []
    for step in prior:
        cap = str(step.get("capability") or "")
        if cap and cap not in caps:
            caps.append(cap)
    for h in feedback.constraints:
        if h not in caps:
            caps.append(h)
    for d in feedback.desired_primitives:
        if d not in caps:
            caps.append(d)
    if error_class == ErrorClass.PLANNING.value:
        return compose_plan(
            attempt.goal,
            caps or [str(s.get("capability") or "") for s in prior if s.get("capability")],
            prior_plan=prior,
            discrepancy=feedback.inferred_problem or feedback.desired_change or "planning error",
        )
    if not caps:
        return compose_plan(
            attempt.goal,
            [],
            prior_plan=prior,
            discrepancy=feedback.inferred_problem or "correction",
        )
    return [{"capability": c, "action": "hold" if str(c).startswith("hold_") else "apply", "goal": attempt.goal} for c in caps]


@dataclass
class CorrectionResult:
    handled: bool
    spoken: str
    feedback: UserFeedback | None
    error_class: str
    retried: bool
    skill_updated: bool
    rsi_escalated: bool
    ask: bool
    plan: list[dict[str, Any]] = field(default_factory=list)
    skill: CompiledSkill | None = None
    evaluation: Any = None
    events: list[str] = field(default_factory=list)


def _spoken(error_class: str, feedback: UserFeedback, retried: bool, validated: bool) -> str:
    if error_class == ErrorClass.VALIDATION.value:
        return "Got it."
    if error_class == ErrorClass.ASK.value:
        return "I can tell that wasn't right — arm position or direction?"
    if error_class == ErrorClass.EXECUTION.value:
        return "Trying that again."
    if error_class == ErrorClass.SYSTEM.value:
        return "I'll try a different way."
    if error_class == ErrorClass.PLANNING.value:
        return "Different sequence this time."
    return "Like this?"


def handle_user_feedback(
    message: str,
    attempts: AttemptStore,
    *,
    skills: SkillStore,
    self_model: SelfModel,
    events: EventLog | None = None,
    rsi: RSIPipeline | None = None,
    executor: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    observer: Callable[[], dict[str, Any]] | None = None,
    interpreter: FeedbackInterpreter | None = None,
    feedback: UserFeedback | None = None,
) -> CorrectionResult:
    log = events or EventLog()
    attempt = attempts.latest(awaiting=True)
    fb = feedback or interpret_feedback(message, attempt, self_model, interpreter=interpreter)
    if fb.feedback_type != FeedbackType.NONE.value and attempt is None:
        attempt = attempts.latest()
    if fb.feedback_type == FeedbackType.NONE.value:
        return CorrectionResult(False, "", None, ErrorClass.ASK.value, False, False, False, False)
    log.emit(
        "feedback_interpreted",
        feedback_type=fb.feedback_type,
        attempt_id=fb.target_attempt_id,
        confidence=fb.confidence,
    )
    error = diagnose(fb, attempt, attempts)
    fb.error_class = error
    log.emit("correction_diagnosed", error_class=error, confidence=fb.confidence)
    if error == ErrorClass.ASK.value or attempt is None:
        log.emit("feedback_asked", reason="insufficient evidence")
        return CorrectionResult(
            True,
            _spoken(ErrorClass.ASK.value, fb, False, False),
            fb,
            ErrorClass.ASK.value,
            False,
            False,
            False,
            True,
            events=log.kinds(),
        )

    attempt.awaiting_feedback = False
    attempt.error_class = error
    attempts.update(attempt)

    if error == ErrorClass.VALIDATION.value:
        skill = None
        if attempt.skill:
            skill = skills.get(attempt.skill) or skills.preferred(attempt.skill)
        if skill is not None:
            skill.validated = True
            skill.successes = int(skill.successes or 0) + 1
            skill.last_validated = _now()
            skill.last_used = _now()
            skills.put(skill)
            log.emit("skill_updated", skill_id=skill.skill_id, reason="positive_validation")
        return CorrectionResult(
            True,
            _spoken(error, fb, False, True),
            fb,
            error,
            False,
            skill is not None,
            False,
            False,
            plan=list(attempt.plan),
            skill=skill,
            events=log.kinds(),
        )

    escalated = False
    if error == ErrorClass.SYSTEM.value and rsi is not None:
        rsi.observe_weakness(
            subsystem="feedback_correction",
            problem="repeated body-motion corrections on the same skill",
            evidence=[fb.user_message, attempt.goal],
            frequency=max(SYSTEMIC_THRESHOLD, attempts.negatives_for(skill=attempt.skill) + 1),
            hypothesis="corrective feedback is not being compiled into the stored procedure",
            proposed_improvement="weight recent validated corrections in skill retrieval",
            expected_benefit="fewer repeated motion corrections",
            risk="low",
            measurable_success_criteria="repeat corrections on the same skill decrease",
        )
        escalated = True

    plan = propose_corrected_plan(attempt, fb, error_class=error)
    sim = _simulate(plan, self_model)
    if not sim.get("safe", True):
        return CorrectionResult(
            True,
            "The corrected motion is not safe to try on this body.",
            fb,
            error,
            False,
            False,
            escalated,
            False,
            plan=plan,
            events=log.kinds(),
        )

    evaluation = None
    retried = False
    if executor is not None:
        before = observer() if observer else {}
        execution = executor(plan) or {}
        after = observer() if observer else dict(before)
        evaluation = evaluate_outcome(
            expected=attempt.goal,
            before=before,
            after=after,
            execution=execution,
        )
        retried = True
        log.emit("correction_retried", attempt_id=attempt.attempt_id, error_class=error)
        attempts.record(
            goal=attempt.goal,
            request=attempt.request,
            skill=attempt.skill,
            plan=plan,
            observations=after,
            outcome={"execution": execution},
            evaluation={
                "success": evaluation.success,
                "discrepancies": evaluation.discrepancies,
                "unexpected_effects": evaluation.unexpected_effects,
            },
            goal_id=attempt.goal_id,
            awaiting_feedback=True,
        )

    new_skill = None
    updated = False
    # Local retry first. Persist a candidate only after repeated same-class misses.
    if error == ErrorClass.SYSTEM.value and attempt.skill:
        base = skills.preferred(attempt.skill) or skills.get(attempt.skill)
        if base is not None:
            new_skill = skills.publish_version(base, plan=plan, source="corrected", validated=False)
            log.emit("skill_updated", skill_id=new_skill.skill_id, reason="corrective_candidate")
            updated = True

    return CorrectionResult(
        True,
        _spoken(error, fb, retried, False),
        fb,
        error,
        retried,
        updated,
        escalated,
        False,
        plan=plan,
        skill=new_skill,
        evaluation=evaluation,
        events=log.kinds(),
    )
