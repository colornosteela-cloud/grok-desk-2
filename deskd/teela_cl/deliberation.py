"""Deliberation throttle: FAST by default; DEEP only when features warrant it.

Mode is chosen from confidence / novelty / risk / ambiguity — not complexity keywords.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .interpreter import phrase_score, record_backed_interpreter
from .records import AttemptRecord, CompiledSkill, InterpretedIntent
from .self_model import SelfModel, snapshot_self_model
from .skill_store import SkillStore


class PolicyInterpreter(Protocol):
    def __call__(self, request: str, skills: list[CompiledSkill], self_model: SelfModel) -> InterpretedIntent: ...


@dataclass
class TurnFeatures:
    """Injectable situation features. Tests set these; production may derive them."""

    confidence: float = 0.85
    ambiguity: float = 0.1
    novelty: float = 0.1
    risk: float = 0.1
    consequence: float = 0.1
    needs_perception: bool = False
    needs_memory: bool = False
    needs_tools: bool = False
    needs_learning: bool = False
    needs_plan: bool = False


@dataclass
class TurnPolicy:
    reasoning_mode: str = "fast"  # fast | normal | deep
    response_mode: str = "natural"  # minimal | natural | explanatory
    confidence: float = 0.85
    ambiguity: float = 0.1
    novelty: float = 0.1
    risk: float = 0.1
    planning_depth: int = 0
    needs_perception: bool = False
    needs_memory: bool = False
    needs_tools: bool = False
    needs_learning: bool = False
    needs_plan: bool = False
    response_budget: str = "0-2 sentences"
    reason: str = "default fast"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InteractionContext:
    active_context: str = ""
    goal: str = ""
    last_attempt_id: str | None = None
    user_feedback_expected: bool = False
    conversation_momentum: str = "talk"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "InteractionContext":
        if not isinstance(raw, dict):
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})


def load_momentum(root: Path | None) -> InteractionContext:
    if root is None:
        return InteractionContext()
    path = Path(root) / "momentum.json"
    if not path.is_file():
        return InteractionContext()
    try:
        return InteractionContext.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError):
        return InteractionContext()


def save_momentum(root: Path | None, ctx: InteractionContext) -> None:
    if root is None:
        return
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    (path / "momentum.json").write_text(json.dumps(ctx.to_dict(), indent=2), encoding="utf-8")


def _token_count(text: str) -> int:
    return len([w for w in (text or "").replace("?", " ").replace(".", " ").split() if w])


def hot_skill_match(text: str, skills: list[CompiledSkill] | None) -> CompiledSkill | None:
    """Cached skill hit from records. No action-name conditionals."""
    best: tuple[float, CompiledSkill | None] = (0.0, None)
    for sk in skills or []:
        if getattr(sk, "kind", "") == "primitive":
            continue
        if not sk.validated:
            continue
        score = phrase_score(text, sk.semantic_description)
        for g in sk.supported_goals or []:
            score = max(score, phrase_score(text, str(g)))
        if score > best[0]:
            best = (score, sk)
    if best[0] >= 0.85:
        return best[1]
    return None


def _features_from_turn(
    text: str,
    *,
    features: TurnFeatures | None,
    awaiting: AttemptRecord | None,
    skills: list[CompiledSkill] | None,
    momentum: InteractionContext | None,
    interpreter: PolicyInterpreter | None,
    self_model: SelfModel | None,
) -> TurnFeatures:
    if features is not None:
        return features
    feat = TurnFeatures()
    t = (text or "").strip()
    n = _token_count(t)
    social = False
    domain = ""
    try:
        model = self_model or snapshot_self_model()
        intent = (interpreter or record_backed_interpreter)(t, list(skills or []), model)
        domain = str(intent.domain or "")
        social = domain == "talk"
    except Exception:
        social = False
    hot = hot_skill_match(t, skills)
    short = n <= 4 and bool(awaiting) and not t.endswith("?")
    recall = bool(
        re.search(
            r"\b(?:my name|your name|who am i|do you (?:know|remember|recall)|"
            r"what(?:'s| is) my|remember (?:my|what))\b",
            t,
            re.I,
        )
    )
    if social:
        feat.needs_memory = recall
        return feat
    if domain == "software":
        feat.needs_tools = True
        feat.needs_learning = False
        feat.needs_plan = False
        feat.confidence = 0.75
        feat.novelty = 0.15
        return feat
    if domain == "perception":
        feat.needs_tools = True
        feat.needs_perception = True
        feat.confidence = 0.7
        return feat
    if short:
        feat.needs_tools = True
        feat.confidence = 0.8
        return feat
    if hot is not None:
        feat.needs_tools = True
        feat.confidence = float(hot.confidence or 0.9)
        feat.novelty = 0.05
        feat.risk = 0.1
        tl = t.lower()
        if n >= 10 or " then " in f" {tl} ":
            feat.ambiguity = 0.5
            feat.needs_plan = True
            feat.needs_perception = True
        return feat
    motor = False
    try:
        import robot_sim

        motor = bool(robot_sim.looks_like_motor(t) or robot_sim.infer_command(t))
    except Exception:
        motor = False
    if not motor:
        try:
            import virtual_body

            motor = virtual_body.teela_args_from_intent(t) is not None
        except Exception:
            pass
    if motor and hot is None:
        feat.novelty = 0.65
        feat.confidence = 0.35
        feat.needs_tools = True
        feat.needs_learning = True
        feat.needs_plan = True
        return feat
    tl = t.lower()
    asking = t.endswith("?") and not tl.startswith(("can you", "could you", "will you", "try"))
    if asking:
        feat.needs_memory = recall
        return feat
    if re.search(r"\b(?:why|explain|details|how come)\b", tl):
        feat.needs_learning = False
        feat.needs_plan = False
        feat.confidence = max(feat.confidence, 0.6)
        return feat
    if not social and not hot and tl.startswith(("can you", "could you", "will you", "try ")):
        feat.novelty = 0.65
        feat.confidence = 0.3
        feat.needs_tools = True
        feat.needs_learning = True
        feat.needs_plan = True
        return feat
    if n >= 12:
        feat.novelty = 0.7
        feat.ambiguity = 0.55
        feat.confidence = 0.3
        feat.needs_plan = True
        feat.needs_tools = True
        return feat
    if momentum and momentum.conversation_momentum == "physical_training" and n <= 4:
        feat.needs_tools = True
        return feat
    return feat


def choose_turn_policy(
    text: str,
    *,
    features: TurnFeatures | None = None,
    awaiting_attempt: AttemptRecord | None = None,
    skills: list[CompiledSkill] | None = None,
    self_model: SelfModel | None = None,
    momentum: InteractionContext | None = None,
    interpreter: PolicyInterpreter | None = None,
) -> TurnPolicy:
    """FAST default. DEEP when injected or derived uncertainty/novelty/risk is high."""
    feat = _features_from_turn(
        text,
        features=features,
        awaiting=awaiting_attempt,
        skills=skills,
        momentum=momentum,
        interpreter=interpreter,
        self_model=self_model,
    )
    mode = "fast"
    depth = 0
    if feat.risk >= 0.5 or feat.novelty >= 0.6 or feat.confidence <= 0.35 or feat.consequence >= 0.5:
        mode = "deep"
        depth = 2
    elif feat.needs_learning or feat.needs_plan or feat.ambiguity >= 0.45:
        mode = "normal"
        depth = 1
    elif feat.needs_tools and feat.confidence < 0.7:
        mode = "normal"
        depth = 1
    response = "natural"
    tl = (text or "").lower()
    if re.search(r"\b(?:why|explain|details|how come)\b", tl):
        response = "explanatory"
        mode = "normal" if mode == "fast" else mode
        feat.needs_learning = False
    elif mode == "deep":
        response = "minimal"
    elif feat.needs_tools and mode == "fast":
        response = "minimal"
    budget = "0-2 sentences"
    if response == "explanatory":
        budget = "as needed"
    reason = {
        "fast": "ordinary conversation or familiar low-risk skill",
        "normal": "some uncertainty or tool need",
        "deep": "novelty, risk, or low confidence",
    }[mode]
    return TurnPolicy(
        reasoning_mode=mode,
        response_mode=response,
        confidence=float(feat.confidence),
        ambiguity=float(feat.ambiguity),
        novelty=float(feat.novelty),
        risk=float(feat.risk),
        planning_depth=depth,
        needs_perception=bool(feat.needs_perception),
        needs_memory=bool(feat.needs_memory),
        needs_tools=bool(feat.needs_tools),
        needs_learning=bool(feat.needs_learning) and mode != "fast",
        needs_plan=bool(feat.needs_plan) and mode != "fast",
        response_budget=budget,
        reason=reason,
    )


def apply_response_budget(text: str | None, policy: TurnPolicy | None) -> str | None:
    """Cap spoken length. Does not turn DEEP into an essay."""
    if not text:
        return text
    if policy is None or policy.response_mode == "explanatory":
        return text
    import re

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    keep = parts[:2]
    out = " ".join(keep).strip()
    return out or text


def strip_internal_labels(text: str | None) -> str | None:
    if not text:
        return text
    import re

    out = re.sub(r"<think\b[^>]*>.*?</think>", " ", text, flags=re.I | re.S)
    out = re.sub(r"</?think\b[^>]*>", " ", out, flags=re.I)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", out) if p.strip()]
    collapsed: list[str] = []
    for part in parts:
        if collapsed and part.lower() == collapsed[-1].lower():
            continue
        collapsed.append(part)
    if collapsed:
        out = " ".join(collapsed)
    out = re.sub(
        r"\b(?:COMPOSE|EXECUTE|LEARNING(?:\s+LOOP)?|LEARN|RESEARCH|RSI|"
        r"capability\s+confidence|planning\s+loop)\b",
        "",
        out,
        flags=re.I,
    )
    out = re.sub(r"\bconfidence\s*[:=]?\s*0\.\d+\b", "", out, flags=re.I)
    out = re.sub(r"\bat\s+0\.\d+\b", "", out, flags=re.I)
    out = re.sub(r"\b0\.\d+\b", "", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([.,!?])", r"\1", out).strip(" ,;")
    return out or None
