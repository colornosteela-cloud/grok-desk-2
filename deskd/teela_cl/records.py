"""Typed records for capability, learning, memory kinds, and RSI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    EXECUTE = "execute"
    COMPOSE = "compose"
    LEARN = "learn"
    RESEARCH = "research"
    OBSERVE = "observe"
    PRACTICE = "practice"
    ASK = "ask"
    REFUSE = "refuse"


class MemoryKind(str, Enum):
    KNOWLEDGE = "knowledge"
    STRATEGY = "strategy"
    SKILL = "skill"
    EXPERIENCE = "experience"
    SELF_IMPROVEMENT = "self_improvement"


class CandidateState(str, Enum):
    PROPOSED = "proposed"
    BUILDING = "building"
    TESTING = "testing"
    FAILED = "failed"
    REGRESSED = "regressed"
    PASSED = "passed"
    AWAITING_PROMOTION = "awaiting_promotion"
    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"


class FeedbackType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"
    PREFERENCE = "preference"
    CLARIFICATION = "clarification"
    NONE = "none"


class ErrorClass(str, Enum):
    EXECUTION = "execution"
    SKILL = "skill"
    PLANNING = "planning"
    MISUNDERSTANDING = "misunderstanding"
    MISSING_CAPABILITY = "missing_capability"
    SYSTEM = "system"
    VALIDATION = "validation"
    ASK = "ask"


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class InterpretedIntent:
    goal: str
    required_capabilities: list[str]
    domain: str
    expected_outcome: str
    confidence: float = 0.5
    notes: str = ""


@dataclass
class CapabilityAssessment:
    goal: str
    interpreted_intent: str
    required_capabilities: list[str]
    matched_skills: list[str]
    available_primitives: list[str]
    missing_capabilities: list[str]
    missing_knowledge: list[str]
    confidence: float
    executable: bool
    composable: bool
    learnable: bool
    decision: str
    reason: str
    # backward-compat aliases used by existing deskd tests
    intent: str = ""
    capability: str | None = None
    known: bool = False
    missing_primitives: list[str] = field(default_factory=list)
    domain: str = ""

    def __post_init__(self) -> None:
        if not self.intent:
            self.intent = self.interpreted_intent or self.goal
        if self.capability is None and self.matched_skills:
            self.capability = self.matched_skills[0]
        if not self.missing_primitives:
            self.missing_primitives = list(self.missing_capabilities)


@dataclass
class CompiledSkill:
    skill_id: str
    name: str
    semantic_description: str
    supported_goals: list[str]
    prerequisites: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    plan: list[dict[str, Any]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    validation_method: str = "outcome_evaluation"
    safety_constraints: list[str] = field(default_factory=list)
    confidence: float = 0.5
    successes: int = 0
    failures: int = 0
    source: str = "seed"
    version: int = 1
    created_at: str = field(default_factory=_now)
    last_used: str | None = None
    last_validated: str | None = None
    kind: str = "skill"  # skill | primitive
    tool: str | None = None
    tool_args: dict[str, Any] | None = None
    validated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CompiledSkill":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        data = {k: v for k, v in raw.items() if k in known}
        if "supported_goals" not in data and raw.get("goals"):
            data["supported_goals"] = list(raw["goals"])
        if "semantic_description" not in data and raw.get("description"):
            data["semantic_description"] = str(raw["description"])
        if "name" not in data:
            data["name"] = str(data.get("skill_id") or "skill")
        return cls(**data)


@dataclass
class OutcomeEvaluation:
    success: bool
    confidence: float
    evidence: dict[str, Any]
    discrepancies: list[str]
    unexpected_effects: list[str]
    recommendation: str


@dataclass
class LearningResult:
    success: bool
    decision: str
    attempts: int
    skill: CompiledSkill | None
    evaluation: OutcomeEvaluation | None
    plan: list[dict[str, Any]]
    reason: str
    events: list[str] = field(default_factory=list)


@dataclass
class KnowledgeRecord:
    knowledge_id: str
    text: str
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)


@dataclass
class StrategyRecord:
    strategy_id: str
    description: str
    applies_to: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)


@dataclass
class ExperienceRecord:
    experience_id: str
    goal: str
    plan: list[dict[str, Any]]
    success: bool
    evidence: dict[str, Any]
    discrepancies: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)


@dataclass
class AttemptRecord:
    attempt_id: str
    goal_id: str
    goal: str
    skill: str | None = None
    plan: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=_now)
    completed_at: str | None = None
    observations: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    awaiting_feedback: bool = True
    request: str = ""
    error_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AttemptRecord":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class UserFeedback:
    feedback_type: str
    target_goal_id: str
    target_attempt_id: str | None
    user_message: str
    inferred_problem: str | None = None
    desired_change: str | None = None
    constraints: list[str] = field(default_factory=list)
    confidence: float = 0.0
    error_class: str = ErrorClass.ASK.value
    desired_primitives: list[str] = field(default_factory=list)


@dataclass
class ImprovementObservation:
    subsystem: str
    problem: str
    evidence: list[str]
    frequency: int
    hypothesis: str
    proposed_improvement: str
    expected_benefit: str
    risk: str
    measurable_success_criteria: str


@dataclass
class LearnerSnapshot:
    version: str
    retrieval_threshold: float = 0.85
    max_attempts: int = 4
    trajectory_weight: float = 0.0
    replay_penalty: float = 1.0
    invariants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LearnerSnapshot":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class ImprovementRecord:
    improvement_id: str
    timestamp: str
    originating_problem: str
    evidence: list[str]
    hypothesis: str
    affected_subsystem: str
    baseline_version: str
    candidate_version: str
    proposed_change: dict[str, Any]
    baseline_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    regression_results: dict[str, Any]
    safety_results: dict[str, Any]
    decision: str
    promoted: bool
    rollback_target: str | None
    post_deployment_metrics: dict[str, float] = field(default_factory=dict)
    state: str = CandidateState.PROPOSED.value
    observation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
