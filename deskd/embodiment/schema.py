#!/usr/bin/env python3
"""Typed body-state and teela_body_action dicts. No HTTP, no LLM."""
from __future__ import annotations

import time
from typing import Any

import robot_sim

JOINT_NAMES: tuple[str, ...] = tuple(robot_sim.JOINTS)
JOINT_LIMITS: dict[str, tuple[float, float]] = dict(robot_sim.JOINTS)
JOINT_ALIASES: dict[str, str] = dict(robot_sim.ALIASES)

LAYERS = ("intended", "expected", "simulated", "observed")

SKILLS: tuple[str, ...] = (
    "look_at",
    "orient_head",
    "orient_torso",
    "reach",
    "retract",
    "grasp",
    "release",
    "lift",
    "place",
    "point",
    "handover",
    "stand",
    "sit",
    "turn",
    "step",
    "walk_to",
    "stabilize",
    "stop",
)

SPEEDS = ("slow", "cautious", "normal", "fast")
EFFECTORS = (
    "head",
    "torso",
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "body",
)

BODY_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["skill"],
    "additionalProperties": False,
    "properties": {
        "skill": {"type": "string", "enum": list(SKILLS)},
        "effector": {"type": "string", "enum": list(EFFECTORS)},
        "target": {"type": ["string", "null"]},
        "speed": {"type": "string", "enum": list(SPEEDS)},
        "constraints": {
            "type": "object",
            "properties": {
                "maintain_balance": {"type": "boolean"},
                "avoid_collision": {"type": "boolean"},
            },
        },
    },
}


def canon_joint(name: str) -> str:
    raw = str(name or "").strip()
    return JOINT_ALIASES.get(raw, raw)


def clamp_joints(raw: Any) -> dict[str, float]:
    out = {name: 0.0 for name in JOINT_NAMES}
    src = raw if isinstance(raw, dict) else {}
    for key, val in src.items():
        name = canon_joint(str(key))
        if name not in robot_sim.JOINTS:
            continue
        clamped = robot_sim._clamp(name, val)
        if clamped is not None:
            out[name] = clamped
    return out


def layer(
    source: str,
    *,
    joints: dict[str, float] | None = None,
    pose: str = "home",
    motion: str = "idle",
    seq: int = 0,
    ts: float | None = None,
) -> dict[str, Any]:
    if source not in LAYERS:
        raise ValueError(f"unknown body layer {source!r}")
    return {
        "source": source,
        "joints": clamp_joints(joints),
        "pose": str(pose or "home"),
        "motion": str(motion or "idle"),
        "seq": int(seq or 0),
        "ts": float(time.time() if ts is None else ts),
    }


def empty_health() -> dict[str, Any]:
    return {
        "joints_ok": True,
        "emergency_stop": False,
        "motors": True,
        "battery": None,
        "hardware": False,
    }


def body_action(
    skill: str,
    *,
    effector: str | None = None,
    target: str | None = None,
    speed: str = "cautious",
    maintain_balance: bool = True,
    avoid_collision: bool = True,
) -> dict[str, Any]:
    skill = str(skill or "").strip().lower()
    if skill not in SKILLS:
        raise ValueError(f"unknown skill {skill!r}")
    spd = str(speed or "cautious").strip().lower()
    if spd not in SPEEDS:
        spd = "cautious"
    out: dict[str, Any] = {"skill": skill, "speed": spd}
    if effector:
        out["effector"] = str(effector)
    if target is not None:
        out["target"] = str(target)
    out["constraints"] = {
        "maintain_balance": bool(maintain_balance),
        "avoid_collision": bool(avoid_collision),
    }
    return out
