#!/usr/bin/env python3
"""Compatibility façade over teela_cl. No action-name decision logic lives here."""

from __future__ import annotations

from typing import Any, Iterable

from teela_cl.records import CapabilityAssessment, CompiledSkill
from teela_cl.resolver import assess_capability as cl_assess, needs_learn_attempt as cl_needs
from teela_cl.skill_store import SkillStore, seed_skills


class Skill:
    """Legacy skill shape used by older tests; maps onto CompiledSkill."""

    def __init__(self, compiled: CompiledSkill) -> None:
        self.skill_id = compiled.skill_id
        self.description = compiled.semantic_description
        self.goals = list(compiled.supported_goals)
        self.primitives = list(compiled.required_capabilities)
        self.confidence = compiled.confidence
        self.validated = compiled.validated
        self.kind = compiled.kind
        self.tool = compiled.tool
        self.tool_args = compiled.tool_args
        self._compiled = compiled


class TeelaState:
    def __init__(self, skills: list[Any], self_model: Any = None) -> None:
        self.skills = skills
        self.self_model = self_model


def default_skills() -> list[Skill]:
    return [Skill(s) for s in seed_skills()]


def default_teela_state() -> TeelaState:
    return TeelaState(default_skills())


def teela_state(skill_ids: Iterable[str] | None = None) -> TeelaState:
    store = SkillStore(seed=True)
    if skill_ids is None:
        return TeelaState([Skill(s) for s in store.all()])
    picked = [Skill(s) for s in store.only(skill_ids).all()]
    if not picked:
        from teela_cl.self_model import snapshot_self_model

        empty = snapshot_self_model(learned_skill_ids=[], tools=[])
        empty.primitives = []
        empty.minios_capabilities = []
        empty.software_tools = []
        return TeelaState([], self_model=empty)
    return TeelaState(picked)


def assess_capability(request: str, state: Any = None) -> CapabilityAssessment:
    return cl_assess(request, state)


def needs_learn_attempt(request: str, state: Any = None) -> bool:
    return cl_needs(request, state)
