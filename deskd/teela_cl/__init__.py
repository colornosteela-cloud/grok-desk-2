"""Bounded continual-learning and gated RSI for Teela.

Public entry points used by deskd and tests. Decision logic contains no
user action names; those live only in skill *records* and self-model facts.
"""

from .events import EventLog, EVENT_TYPES
from .records import (
    AttemptRecord,
    CapabilityAssessment,
    CompiledSkill,
    ErrorClass,
    ExperienceRecord,
    FeedbackType,
    ImprovementObservation,
    ImprovementRecord,
    InterpretedIntent,
    KnowledgeRecord,
    LearnerSnapshot,
    LearningResult,
    OutcomeEvaluation,
    StrategyRecord,
    UserFeedback,
)
from .self_model import SelfModel, snapshot_self_model
from .skill_store import SkillStore
from .memory_kinds import TypedMemory
from .interpreter import GoalInterpreter, is_performance_request, record_backed_interpreter
from .resolver import assess_capability, needs_learn_attempt
from .outcome import classify_evidence, evaluate_outcome, is_practice_note
from .learning import LearnContext, compose_plan, learn_goal
from .attempts import AttemptStore
from .feedback import handle_user_feedback, interpret_feedback
from .deliberation import (
    InteractionContext,
    TurnFeatures,
    TurnPolicy,
    apply_response_budget,
    choose_turn_policy,
    strip_internal_labels,
)
from .rsi import (
    PROTECTED_INVARIANTS,
    RSIPipeline,
    CandidateState,
)

__all__ = [
    "EVENT_TYPES",
    "EventLog",
    "AttemptRecord",
    "AttemptStore",
    "CapabilityAssessment",
    "ErrorClass",
    "FeedbackType",
    "UserFeedback",
    "handle_user_feedback",
    "interpret_feedback",
    "TurnFeatures",
    "TurnPolicy",
    "InteractionContext",
    "choose_turn_policy",
    "apply_response_budget",
    "strip_internal_labels",
    "CompiledSkill",
    "ExperienceRecord",
    "ImprovementObservation",
    "ImprovementRecord",
    "InterpretedIntent",
    "KnowledgeRecord",
    "LearnerSnapshot",
    "LearningResult",
    "OutcomeEvaluation",
    "StrategyRecord",
    "SelfModel",
    "snapshot_self_model",
    "SkillStore",
    "TypedMemory",
    "GoalInterpreter",
    "is_performance_request",
    "record_backed_interpreter",
    "assess_capability",
    "needs_learn_attempt",
    "classify_evidence",
    "evaluate_outcome",
    "is_practice_note",
    "LearnContext",
    "compose_plan",
    "learn_goal",
    "PROTECTED_INVARIANTS",
    "RSIPipeline",
    "CandidateState",
]
