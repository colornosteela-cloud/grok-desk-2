"""Outcome evaluation: did the requested result occur, not merely did code run."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .events import EventLog
from .records import OutcomeEvaluation

_WORD = re.compile(r"[a-z0-9]+")
_LEARN_MD = re.compile(r"learn-[a-z0-9-]+\.md$", re.I)
_DOMAIN_FIELDS = (
    ("service_health", ("service", "health", "healthy", "restart")),
    ("ui_state", ("ui", "interface")),
    ("tests_passed", ("test", "tests")),
)


class Observation(Protocol):
    def snapshot(self) -> dict[str, Any]: ...


@dataclass
class ClassifiedEvidence:
    process: list[str] = field(default_factory=list)
    outcome: list[str] = field(default_factory=list)
    discrepancies: list[str] = field(default_factory=list)


def is_practice_note(path: Any, execution: dict[str, Any] | None = None) -> bool:
    """True for executor-written Desktop/learn-*.md notes (process, not outcome)."""
    s = str(path or "").replace("\\", "/")
    if not s:
        return False
    name = s.rsplit("/", 1)[-1]
    if _LEARN_MD.search(name) or "/desktop/learn-" in s.lower():
        return True
    ex = execution or {}
    for key in ("artifact", "desktop_file"):
        v = str(ex.get(key) or "").replace("\\", "/")
        if v and (s == v or s.endswith(v) or v.endswith(name)):
            return True
    return False


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _feedback_affirms(feedback: Any, expected: str) -> bool:
    if feedback is True:
        return True
    if not isinstance(feedback, str) or not feedback.strip():
        return False
    fl = feedback.lower()
    if any(w in fl for w in ("yes", "success", "correct", "worked", "good")):
        return True
    want = {t for t in _tokens(expected) if len(t) > 3}
    return bool(want) and bool(want & _tokens(fl))


def classify_evidence(
    expected: str,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    execution: dict[str, Any] | None = None,
) -> ClassifiedEvidence:
    """Split process evidence (ran/applied/pose/practice note) from outcome evidence.

    Process-only never succeeds. Outcome must be independent of the executor:
    trajectory_ok, affirming user_feedback, a non-practice artifact, a domain
    field named by `expected`, or an explicit goal_met/outcome flag.
    """
    b = dict(before or {})
    a = dict(after or {})
    ex = dict(execution or {})
    process: list[str] = []
    outcome: list[str] = []

    if ex.get("ok") or ex.get("ran") is True or ex.get("success"):
        process.append("execution_ran")
    if a.get("applied_capabilities"):
        process.append("applied_capabilities")
    if a.get("pose") and a.get("pose") != b.get("pose"):
        process.append("pose_change")
    if a.get("joints") and a.get("joints") != b.get("joints"):
        process.append("joint_delta")
    if a.get("motion") and a.get("motion") != b.get("motion"):
        process.append("motion_change")

    art = a.get("artifact")
    if art:
        if is_practice_note(art, ex):
            process.append("practice_note")
        elif art != b.get("artifact"):
            outcome.append("artifact")

    if a.get("trajectory_ok") is True:
        outcome.append("trajectory_ok")
    if a.get("goal_met") is True:
        outcome.append("goal_met")
    if a.get("outcome") is True:
        outcome.append("outcome")
    if _feedback_affirms(a.get("user_feedback"), expected):
        outcome.append("user_feedback")

    perc = a.get("perception")
    if perc is True or (isinstance(perc, dict) and perc.get("goal_met") is True):
        outcome.append("perception")

    exp = (expected or "").lower()
    exp_tokens = _tokens(exp)
    for key, aliases in _DOMAIN_FIELDS:
        if a.get(key) in {None, "", b.get(key)}:
            continue
        improved = key == "service_health" and a.get(key) == "ok" and b.get(key) != "ok"
        improved = improved or (key != "service_health" and a.get(key) != b.get(key) and a.get(key) not in {None, False, "down", "fail"})
        if not improved:
            continue
        names = set(aliases) | set(_tokens(key.replace("_", " ")))
        if names & exp_tokens or any(n in exp for n in names):
            outcome.append(key)

    discrepancies: list[str] = []
    if not outcome:
        discrepancies.append("no evidence the intended outcome occurred")
    return ClassifiedEvidence(process=process, outcome=outcome, discrepancies=discrepancies)


def evaluate_outcome(
    *,
    expected: str,
    before: dict[str, Any],
    after: dict[str, Any],
    execution: dict[str, Any] | None = None,
    assertions: list[Callable[[dict[str, Any], dict[str, Any]], str | None]] | None = None,
    events: EventLog | None = None,
    trajectory_weight: float = 0.0,
) -> OutcomeEvaluation:
    """Compare pre/post observations against the intended outcome.

    `assertions` may add discrepancies; they cannot turn process-only into success.
    A zero-exit / ok:true execution is never sufficient alone.
    """
    classified = classify_evidence(expected, before, after, execution)
    discrepancies: list[str] = list(classified.discrepancies)
    unexpected: list[str] = []
    evidence: dict[str, Any] = {
        "before": before,
        "after": after,
        "execution": execution or {},
        "expected": expected,
        "process": list(classified.process),
        "outcome": list(classified.outcome),
    }
    for fn in assertions or []:
        msg = fn(before, after)
        if msg:
            discrepancies.append(msg)
    if float(trajectory_weight or 0) > 0:
        if after.get("trajectory_ok") is True:
            pass
        elif after.get("trajectory_ok") is False:
            discrepancies.append("trajectory did not match the intended motion")
        elif after.get("pose") == before.get("pose") and not after.get("goal_met"):
            discrepancies.append("no trajectory evidence for the intended outcome")
    if after.get("collision") and not before.get("collision"):
        unexpected.append("collision")
    if after.get("fell") and not before.get("fell"):
        unexpected.append("loss of balance")
    if unexpected:
        discrepancies.extend(f"unexpected: {u}" for u in unexpected)
    success = bool(classified.outcome) and not discrepancies
    conf = 0.9 if success else 0.2
    if unexpected:
        conf = min(conf, 0.5)
    rec = "store_skill" if success else "revise_plan"
    ev = OutcomeEvaluation(
        success=success,
        confidence=conf,
        evidence=evidence,
        discrepancies=discrepancies,
        unexpected_effects=unexpected,
        recommendation=rec,
    )
    if events is not None:
        events.emit(
            "outcome_evaluated",
            success=success,
            discrepancies=discrepancies,
            recommendation=rec,
        )
    return ev
