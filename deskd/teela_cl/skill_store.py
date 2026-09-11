"""Persistent skill registry. Independent of conversation wording."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .records import CompiledSkill, _now


def seed_skills() -> list[CompiledSkill]:
    """Seed records. Paraphrases live in data, not in decision logic."""
    def s(**kw: Any) -> CompiledSkill:
        return CompiledSkill(**kw)

    return [
        s(
            skill_id="gesture.wave",
            name="wave",
            semantic_description="Wave one hand toward a person as a greeting",
            supported_goals=[
                "wave",
                "wave at me",
                "give me a wave",
                "say hello physically",
                "say hello with your hand",
                "say hey with your hand",
                "can u wave",
                "gimme a wave",
                "give me a wave",
                "greet that person",
                "greet me with your hand",
                "greet that person physically",
                "wave your right hand",
                "give me a little greeting gesture",
            ],
            required_capabilities=["raise_arm", "bend_elbow", "oscillate_wrist", "lower_arm"],
            plan=[{"tool": "bot_desktop__teela_body_action", "args": {"skill": "wave", "side": "right"}}],
            expected_outcome="One hand waves toward the person",
            confidence=0.97,
            successes=38,
            failures=1,
            source="seed",
            validated=True,
            last_validated=_now(),
            tool="bot_desktop__teela_body_action",
            tool_args={"skill": "wave", "side": "right"},
            kind="skill",
        ),
        s(
            skill_id="locomotion.walk",
            name="walk",
            semantic_description="Walk in a direction",
            supported_goals=["walk", "walk left", "walk right", "walk forward", "start walking"],
            required_capabilities=["stand", "balance", "step"],
            plan=[{"tool": "bot_desktop__robot_motion", "args": {"cmd": "walk"}}],
            expected_outcome="The body walks",
            confidence=0.95,
            validated=True,
            tool="bot_desktop__robot_motion",
            tool_args={"cmd": "walk"},
        ),
        s(
            skill_id="locomotion.stop",
            name="stop",
            semantic_description="Stop moving and hold still",
            supported_goals=["stop", "stop moving", "halt", "hold still"],
            required_capabilities=["stand"],
            plan=[{"tool": "bot_desktop__teela_stop", "args": {"skill": "stop"}}],
            expected_outcome="Motion stops",
            confidence=0.99,
            validated=True,
            tool="bot_desktop__teela_stop",
            tool_args={"skill": "stop"},
        ),
        s(
            skill_id="gesture.point",
            name="point",
            semantic_description="Point a hand toward a person or object",
            supported_goals=[
                "point",
                "point at",
                "point at the person",
                "point at that",
                "point over there",
            ],
            required_capabilities=["raise_arm"],
            plan=[{"tool": "bot_desktop__teela_gesture", "args": {"gesture": "point", "side": "right"}}],
            expected_outcome="A hand points toward the indicated target",
            confidence=0.9,
            validated=True,
            tool="bot_desktop__teela_gesture",
            tool_args={"gesture": "point", "side": "right"},
            kind="skill",
        ),
        s(
            skill_id="gaze.look",
            name="look",
            semantic_description="Turn the head to look at a person or toward a side",
            supported_goals=[
                "look at me",
                "look left",
                "look right",
                "turn your head left",
                "turn your head right",
                "look at me please",
            ],
            required_capabilities=["orient_head"],
            plan=[{"tool": "bot_desktop__teela_body_action", "args": {"skill": "orient_head"}}],
            expected_outcome="Head orients toward the requested direction",
            confidence=0.96,
            validated=True,
            tool="bot_desktop__teela_body_action",
            tool_args={"skill": "orient_head"},
            kind="skill",
        ),
        s(
            skill_id="pose.raise_arm",
            name="raise arm",
            semantic_description="Raise one arm overhead",
            supported_goals=[
                "raise your arm",
                "raise arm",
                "put your arm up",
                "arm up",
                "lift your arm",
                "raise your right arm",
                "raise your left arm",
            ],
            required_capabilities=["raise_arm"],
            plan=[{"tool": "bot_desktop__teela_body_action", "args": {"skill": "raise_arm", "side": "right"}}],
            expected_outcome="One arm is raised",
            confidence=0.95,
            validated=True,
            tool="bot_desktop__teela_body_action",
            tool_args={"skill": "raise_arm", "side": "right"},
            kind="skill",
        ),
        s(
            skill_id="pose.home",
            name="home",
            semantic_description="Stand in a neutral home pose",
            supported_goals=[
                "stand",
                "stand still",
                "neutral pose",
                "home pose",
                "put your leg down",
                "lower your leg",
                "leg down",
            ],
            required_capabilities=["stand", "balance"],
            plan=[{"tool": "bot_desktop__robot_pose", "args": {"pose": "home"}}],
            expected_outcome="Neutral standing pose",
            confidence=0.95,
            validated=True,
            tool="bot_desktop__robot_pose",
            tool_args={"pose": "home"},
        ),
    ] + [
        CompiledSkill(
            skill_id=f"primitive.{pid}",
            name=pid,
            semantic_description=desc,
            supported_goals=goals,
            required_capabilities=[],
            kind="primitive",
            validated=True,
            confidence=0.9,
            source="seed",
        )
        for pid, desc, goals in (
            ("balance", "Keep balance while standing", ["balance", "keep balance"]),
            ("weight_shift", "Shift weight from one foot to the other", ["shift weight", "weight shift"]),
            ("foot_slide", "Slide a foot along the ground", ["slide foot", "foot slide"]),
            ("heel_raise", "Raise a heel off the ground", ["raise heel", "heel raise"]),
            ("raise_arm", "Raise an arm", ["raise arm"]),
            ("bend_elbow", "Bend an elbow", ["bend elbow"]),
            ("oscillate_wrist", "Rock a wrist", ["oscillate wrist", "rock wrist"]),
            ("lower_arm", "Lower an arm", ["lower arm"]),
            ("stand", "Stand upright", ["stand"]),
            ("step", "Take a step", ["step"]),
            ("hold_shoulder", "Keep the upper arm or shoulder still", ["hold shoulder", "upper arm still"]),
            ("hold_elbow", "Keep the elbow still", ["hold elbow", "elbow still"]),
            ("orient_head", "Turn or tilt the head to look", ["look left", "look right", "look at me"]),
        )
    ]


class SkillStore:
    def __init__(self, root: Path | None = None, *, seed: bool = True) -> None:
        self.root = Path(root) if root is not None else None
        self.path = (self.root / "skills.json") if self.root is not None else None
        self._skills: dict[str, CompiledSkill] = {}
        if seed:
            for sk in seed_skills():
                self._skills[sk.skill_id] = sk
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        rows = raw if isinstance(raw, list) else raw.get("skills") or []
        for row in rows:
            if not isinstance(row, dict):
                continue
            sk = CompiledSkill.from_dict(row)
            self._skills[sk.skill_id] = sk

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [s.to_dict() for s in self._skills.values()]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def all(self) -> list[CompiledSkill]:
        return list(self._skills.values())

    def get(self, skill_id: str) -> CompiledSkill | None:
        return self._skills.get(skill_id)

    def put(self, skill: CompiledSkill, *, persist: bool = True) -> CompiledSkill:
        self._skills[skill.skill_id] = skill
        if persist:
            self.save()
        return skill

    def preferred(self, skill_id: str) -> CompiledSkill | None:
        sid = str(skill_id or "")
        if not sid:
            return None
        root = sid.split(".v")[0]
        cands: list[CompiledSkill] = []
        for s in self._skills.values():
            if s.kind == "primitive":
                continue
            base = s.skill_id.split(".v")[0]
            if s.skill_id == sid or base == root or s.name == sid or s.name == root.rsplit(".", 1)[-1]:
                cands.append(s)
        if not cands:
            return self._skills.get(sid)
        validated = [s for s in cands if s.validated]
        pool = validated or cands
        return max(pool, key=lambda s: (int(s.version or 1), s.skill_id))

    def publish_version(
        self,
        base: CompiledSkill,
        *,
        plan: list,
        source: str = "corrected",
        validated: bool = False,
    ) -> CompiledSkill:
        from dataclasses import replace

        root = base.skill_id.split(".v")[0]
        ver = int(base.version or 1) + 1
        caps = [str(s.get("capability") or "") for s in plan if isinstance(s, dict) and s.get("capability")]
        nxt = replace(
            base,
            skill_id=f"{root}.v{ver}",
            version=ver,
            plan=list(plan),
            required_capabilities=caps or list(base.required_capabilities),
            source=source,
            validated=validated,
            successes=0,
            failures=0,
            last_validated=None,
        )
        return self.put(nxt)

    def only(self, ids: Iterable[str]) -> "SkillStore":
        wanted = {str(x) for x in ids}
        out = SkillStore(root=None, seed=False)
        for sk in self._skills.values():
            tail = sk.skill_id.rsplit(".", 1)[-1]
            if sk.skill_id in wanted or tail in wanted or sk.name in wanted:
                out._skills[sk.skill_id] = sk
        return out
