#!/usr/bin/env python3
"""Server-side Teela robot body state. Mirrors ui/robot-simulator.html."""
from __future__ import annotations

import math
import re
import time
from typing import Any

JOINTS: dict[str, tuple[float, float]] = {
    "neck_pan": (-70, 70),
    "neck_tilt": (-40, 40),
    "left_shoulder": (-35, 155),
    "left_shoulder_out": (0, 110),
    "left_elbow": (0, 135),
    "left_wrist": (-180, 180),
    "right_shoulder": (-35, 155),
    "right_shoulder_out": (0, 110),
    "right_elbow": (0, 135),
    "right_wrist": (-180, 180),
    "upper_back_pitch": (-30, 55),
    "lower_back_pitch": (-25, 35),
    "lower_back_roll": (-25, 25),
    "left_hip": (-45, 90),
    "left_hip_out": (0, 20),
    "left_knee": (0, 120),
    "left_ankle": (-40, 40),
    "right_hip": (-45, 90),
    "right_hip_out": (0, 20),
    "right_knee": (0, 120),
    "right_ankle": (-40, 40),
}

ALIASES = {
    "neck_yaw": "neck_pan",
    "neck_pitch": "neck_tilt",
    "torso_yaw": "lower_back_roll",
    "left_arm_out": "left_shoulder_out",
    "right_arm_out": "right_shoulder_out",
    "left_leg_out": "left_hip_out",
    "right_leg_out": "right_hip_out",
}

POSES: dict[str, dict[str, float]] = {
    "home": {
        "neck_pan": 0, "neck_tilt": 0, "left_shoulder": 0, "left_shoulder_out": 0, "left_elbow": 0, "left_wrist": 0,
        "right_shoulder": 0, "right_shoulder_out": 0, "right_elbow": 0, "right_wrist": 0, "upper_back_pitch": 0,
        "lower_back_roll": 0, "left_hip": 0, "left_knee": 0, "left_ankle": 0, "right_hip": 0,
        "right_knee": 0, "right_ankle": 0,
    },
    "neutral": {
        "neck_pan": 0, "neck_tilt": 0, "left_shoulder": 0, "left_shoulder_out": 0, "left_elbow": 4, "left_wrist": 0,
        "right_shoulder": 0, "right_shoulder_out": 0, "right_elbow": 4, "right_wrist": 0, "upper_back_pitch": 0,
        "lower_back_roll": 0, "left_hip": 0, "left_knee": 3, "left_ankle": 0, "right_hip": 0,
        "right_knee": 3, "right_ankle": 0,
    },
    "relaxed": {
        "neck_pan": 6, "neck_tilt": 2, "left_shoulder": -5, "left_elbow": 14, "left_wrist": 4,
        "right_shoulder": 4, "right_elbow": 10, "right_wrist": -3, "upper_back_pitch": 2,
        "lower_back_roll": 3, "left_hip": -2, "left_knee": 5, "left_ankle": 1, "right_hip": 4,
        "right_knee": 12, "right_ankle": -3,
    },
    "attention": {
        "neck_pan": 0, "neck_tilt": 0, "left_shoulder": 0, "left_elbow": 0, "left_wrist": 0,
        "right_shoulder": 0, "right_elbow": 0, "right_wrist": 0, "upper_back_pitch": 0,
        "lower_back_roll": 0, "left_hip": 0, "left_knee": 0, "left_ankle": 0, "right_hip": 0,
        "right_knee": 0, "right_ankle": 0,
    },
    "ready": {
        "neck_pan": 0, "neck_tilt": 0, "left_shoulder": 24, "left_elbow": 42, "left_wrist": 8,
        "right_shoulder": 24, "right_elbow": 42, "right_wrist": -8, "upper_back_pitch": 3,
        "lower_back_roll": 0, "left_hip": 12, "left_knee": 22, "left_ankle": -5, "right_hip": 12,
        "right_knee": 22, "right_ankle": -5,
    },
    "sit": {
        "neck_pan": 0, "neck_tilt": 4, "left_shoulder": 8, "left_elbow": 38, "left_wrist": 0,
        "right_shoulder": 8, "right_elbow": 38, "right_wrist": 0, "upper_back_pitch": 8,
        "lower_back_roll": 0, "left_hip": 58, "left_knee": 92, "left_ankle": 10, "right_hip": 58,
        "right_knee": 92, "right_ankle": 10,
    },
    "bow": {
        "neck_pan": 0, "neck_tilt": -16, "left_shoulder": 6, "left_elbow": 12, "left_wrist": 4,
        "right_shoulder": 6, "right_elbow": 12, "right_wrist": -4, "upper_back_pitch": 30,
        "lower_back_roll": 0, "left_hip": 14, "left_knee": 12, "left_ankle": 8, "right_hip": 14,
        "right_knee": 12, "right_ankle": 8,
    },
    "lean_left": {
        "neck_pan": 0, "neck_tilt": 0, "left_shoulder": -8, "left_elbow": 8, "left_wrist": 0,
        "right_shoulder": 8, "right_elbow": 8, "right_wrist": 0, "upper_back_pitch": 1,
        "lower_back_roll": -12, "left_hip": 0, "left_knee": 4, "left_ankle": 2, "right_hip": 0,
        "right_knee": 8, "right_ankle": -2,
    },
    "lean_right": {
        "neck_pan": 0, "neck_tilt": 0, "left_shoulder": 8, "left_elbow": 8, "left_wrist": 0,
        "right_shoulder": -8, "right_elbow": 8, "right_wrist": 0, "upper_back_pitch": 1,
        "lower_back_roll": 12, "left_hip": 0, "left_knee": 8, "left_ankle": -2, "right_hip": 0,
        "right_knee": 4, "right_ankle": 2,
    },
    "hands_up": {
        "neck_pan": 0, "neck_tilt": -4, "left_shoulder": 145, "left_elbow": 18, "left_wrist": 0,
        "right_shoulder": 145, "right_elbow": 18, "right_wrist": 0, "upper_back_pitch": -3,
        "lower_back_roll": 0, "left_hip": 0, "left_knee": 4, "left_ankle": 0, "right_hip": 0,
        "right_knee": 4, "right_ankle": 0,
    },
    "shrug": {
        "neck_pan": 0, "neck_tilt": 2, "left_shoulder": 48, "left_elbow": 72, "left_wrist": 10,
        "right_shoulder": 48, "right_elbow": 72, "right_wrist": -10, "upper_back_pitch": 4,
        "lower_back_roll": 0, "left_hip": 0, "left_knee": 4, "left_ankle": 0, "right_hip": 0,
        "right_knee": 4, "right_ankle": 0,
    },
    "tpose": {
        "neck_pan": 0, "neck_tilt": 0, "left_shoulder": 0, "left_shoulder_out": 90, "left_elbow": 0, "left_wrist": 0,
        "right_shoulder": 0, "right_shoulder_out": 90, "right_elbow": 0, "right_wrist": 0, "upper_back_pitch": 0,
        "lower_back_roll": 0, "left_hip": 0, "left_knee": 0, "left_ankle": 0, "right_hip": 0,
        "right_knee": 0, "right_ankle": 0,
    },
    "wave": {
        "neck_pan": 0, "neck_tilt": 0, "left_shoulder": 0, "left_elbow": 5, "left_wrist": 0,
        "right_shoulder": 24, "right_shoulder_out": 0, "right_elbow": 118, "right_wrist": 0, "upper_back_pitch": 0,
        "lower_back_roll": 0, "left_hip": 0, "left_knee": 4, "left_ankle": 0, "right_hip": 0,
        "right_knee": 4, "right_ankle": 0,
    },
    "arms_forward": {
        "neck_pan": 0, "neck_tilt": 0, "left_shoulder": 78, "left_elbow": 8, "left_wrist": 0,
        "right_shoulder": 78, "right_elbow": 8, "right_wrist": 0, "upper_back_pitch": 0,
        "lower_back_roll": 0, "left_hip": 0, "left_knee": 6, "left_ankle": 0, "right_hip": 0,
        "right_knee": 6, "right_ankle": 0,
    },
    "bend_forward": {
        "neck_pan": 0, "neck_tilt": 10, "left_shoulder": 0, "left_elbow": 5, "left_wrist": 0,
        "right_shoulder": 0, "right_elbow": 5, "right_wrist": 0, "upper_back_pitch": 40,
        "lower_back_pitch": 22, "lower_back_roll": 0, "left_hip": 8, "left_knee": 10, "left_ankle": 0, "right_hip": 8,
        "right_knee": 10, "right_ankle": 0,
    },
    "ground_support": {
        "neck_pan": 0, "neck_tilt": 14, "left_shoulder": 82, "left_elbow": 28, "left_wrist": 0,
        "right_shoulder": 82, "right_elbow": 28, "right_wrist": 0, "upper_back_pitch": 48,
        "lower_back_pitch": 28, "lower_back_roll": 0, "left_hip": 18, "left_knee": 14, "left_ankle": 8, "right_hip": 18,
        "right_knee": 14, "right_ankle": 8,
    },
    "left_leg_out": {
        "neck_pan": 0, "neck_tilt": 0, "left_shoulder": 0, "left_elbow": 5, "left_wrist": 0,
        "right_shoulder": 0, "right_elbow": 5, "right_wrist": 0, "upper_back_pitch": 0,
        "lower_back_roll": 0, "left_hip": 0, "left_hip_out": 20, "left_knee": 6, "left_ankle": 0, "right_hip": 0,
        "right_hip_out": 0, "right_knee": 6, "right_ankle": 0,
    },
    "right_leg_out": {
        "neck_pan": 0, "neck_tilt": 0, "left_shoulder": 0, "left_elbow": 5, "left_wrist": 0,
        "right_shoulder": 0, "right_elbow": 5, "right_wrist": 0, "upper_back_pitch": 0,
        "lower_back_roll": 0, "left_hip": 0, "left_hip_out": 0, "left_knee": 6, "left_ankle": 0, "right_hip": 0,
        "right_hip_out": 20, "right_knee": 6, "right_ankle": 0,
    },
    "left_leg_raise": {
        "neck_pan": 0, "neck_tilt": 0, "left_shoulder": 0, "left_elbow": 5, "left_wrist": 0,
        "right_shoulder": 0, "right_elbow": 5, "right_wrist": 0, "upper_back_pitch": 0,
        "lower_back_roll": 0, "left_hip": 90, "left_knee": 90, "left_ankle": 0, "right_hip": 0,
        "right_knee": 6, "right_ankle": 0,
    },
    "right_leg_raise": {
        "neck_pan": 0, "neck_tilt": 0, "left_shoulder": 0, "left_elbow": 5, "left_wrist": 0,
        "right_shoulder": 0, "right_elbow": 5, "right_wrist": 0, "upper_back_pitch": 0,
        "lower_back_roll": 0, "left_hip": 0, "left_knee": 6, "left_ankle": 0, "right_hip": 90,
        "right_knee": 90, "right_ankle": 0,
    },
    "kneel_right": {
        "neck_pan": 0, "neck_tilt": 2, "left_shoulder": 16, "left_elbow": 22, "left_wrist": 0,
        "right_shoulder": 16, "right_elbow": 22, "right_wrist": 0, "upper_back_pitch": 4,
        "lower_back_pitch": 8, "lower_back_roll": 6, "left_hip": 90, "left_hip_out": 0, "left_knee": 63, "left_ankle": 25,
        "right_hip": 0, "right_hip_out": 0, "right_knee": 100, "right_ankle": 40,
    },
    "kneel_left": {
        "neck_pan": 0, "neck_tilt": 2, "left_shoulder": 16, "left_elbow": 22, "left_wrist": 0,
        "right_shoulder": 16, "right_elbow": 22, "right_wrist": 0, "upper_back_pitch": 4,
        "lower_back_pitch": 8, "lower_back_roll": -6, "left_hip": 0, "left_hip_out": 0, "left_knee": 100, "left_ankle": 40,
        "right_hip": 90, "right_hip_out": 0, "right_knee": 63, "right_ankle": 25,
    },
    "kneel_both": {
        "neck_pan": 0, "neck_tilt": 2, "left_shoulder": 16, "left_elbow": 22, "left_wrist": 0,
        "right_shoulder": 16, "right_elbow": 22, "right_wrist": 0, "upper_back_pitch": 4,
        "lower_back_pitch": 8, "lower_back_roll": 0, "left_hip": 0, "left_hip_out": 0, "left_knee": 100, "left_ankle": 40,
        "right_hip": 0, "right_hip_out": 0, "right_knee": 100, "right_ankle": 40,
    },
    "squat": {
        "neck_pan": 0, "neck_tilt": 6, "left_shoulder": 28, "left_elbow": 22, "left_wrist": 0,
        "right_shoulder": 28, "right_elbow": 22, "right_wrist": 0, "upper_back_pitch": 8,
        "lower_back_roll": 0, "left_hip": 48, "left_knee": 78, "left_ankle": 12, "right_hip": 48,
        "right_knee": 78, "right_ankle": 12,
    },
}

MOTIONS = {"walk", "stop", "demo", "reset", "estop_on", "estop_off", "motors_on", "motors_off"}

_NEG = re.compile(r"\b(?:don't|do not|dont|never|without|can't|cannot|can not)\b", re.IGNORECASE)
_WANT_MOVE = re.compile(
    r"\b(?:i want you to|i want|instead|rather)\b.{0,48}\b(?:walk|wave|look|turn|raise|stop|move|sit|bow)\b",
    re.IGNORECASE,
)


def _blocked_by_negation(text: str) -> bool:
    """True when the whole line is a refusal. 'Don't look, walk' is still a move."""
    t = text or ""
    if not _NEG.search(t):
        return False
    if _WANT_MOVE.search(t):
        return False
    if re.search(r"\b(?:don't|do not|dont)\b.+\b(?:i want|instead|rather)\b", t, re.I):
        return False
    return True
_STOP_MOTION = re.compile(
    r"(?:"
    r"\b(?:stop|halt)\s+(?:walking|walk(?:ing)?(?:\s+demo)?|moving|the\s+(?:walk|motion)|that)|"
    r"\b(?:quit|end|cancel)\s+walk(?:ing)?|"
    r"\b(?:stop|halt|quit|end|cancel)\s+(?:the\s+)?(?:wave|waving|waive|waiving)\b|"
    r"\b(?:not|aren't|isn'?t)\s+(?:wave|waving|waive|waiving)\b|"
    r"\bstop\s+(?:it|this|now|moving|doing(?:\s+that)?|the\s+(?:action|pose|gesture|motion))\b|"
    r"\b(?:go\s+back\s+to|return\s+to|back\s+to)\s+(?:neutral|rest|normal|home|idle)\b|"
    r"\b(?:stand\s+down|at\s+ease|that'?s\s+enough)\b|"
    r"\bneutral(?:\s+(?:pose|position))?\b|"
    r"\bstand\s+still\b|"
    r"\bhold\s+still\b|"
    r"\b(?:can|could|would|will)\s+you\s+stop\b|"
    r"\byou\s+can\s+stop\b|"
    r"^(?:please\s+)?(?:stop|halt)(?:\s+please)?$"
    r")",
    re.IGNORECASE,
)
_NUM = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:deg(?:rees?)?|°)?", re.IGNORECASE)
# Reach-in-front vs raise-overhead. "reach up" / "reach for the sky" stay raises.
_FWD_RE = re.compile(
    r"\b(?:forward|forwards|ahead|in front|"
    r"reach(?:ing)?(?:\s+out)?\s+(?:forward|ahead|in front|toward me|towards me|to me))\b",
    re.IGNORECASE,
)
_RAISE_RE = re.compile(
    r"\b(?:raise|lift|overhead|upward)\b|"
    r"\b(?:hands?|arms?)\s+up\b|"
    r"\breach(?:ing)?\s+(?:up|for the (?:sky|ceiling))\b",
    re.IGNORECASE,
)

# Longer phrases first so "lean right" wins over a generic "right".
_POSE_PHRASES: list[tuple[str, str]] = sorted(
    [
        ("hands up", "hands_up"),
        ("arms up", "hands_up"),
        ("both hands up", "hands_up"),
        ("both arms up", "hands_up"),
        ("put your hands up", "hands_up"),
        ("put your arms up", "hands_up"),
        ("raise your hands", "hands_up"),
        ("raise both hands", "hands_up"),
        ("raise both of your hands", "hands_up"),
        ("raise your arms", "hands_up"),
        ("raise both arms", "hands_up"),
        ("raise both of your arms", "hands_up"),
        ("lift your hands", "hands_up"),
        ("lift your arms", "hands_up"),
        ("hands in the air", "hands_up"),
        ("arms in the air", "hands_up"),
        ("take a bow", "bow"),
        ("bow", "bow"),
        ("shrug", "shrug"),
        ("hands down", "home"),
        ("arms down", "home"),
        ("put your hands down", "home"),
        ("lower your hands", "home"),
        ("stand up straight", "home"),
        ("stand straight", "home"),
        ("straighten up", "home"),
        ("stand up", "home"),
        ("stand", "home"),
        ("go home", "home"),
        ("home pose", "home"),
        ("sit down", "sit"),
        ("sit", "sit"),
        ("t-pose", "tpose"),
        ("t pose", "tpose"),
        ("tpose", "tpose"),
        ("arms forward", "arms_forward"),
        ("arms out in front", "arms_forward"),
        ("both arms forward", "arms_forward"),
        ("reach forward", "arms_forward"),
        ("bend forward", "bend_forward"),
        ("touch the ground", "ground_support"),
        ("touch the floor", "ground_support"),
        ("hands on the floor", "ground_support"),
        ("ground support", "ground_support"),
        ("left 90", "left_leg_raise"),
        ("right 90", "right_leg_raise"),
        ("raise your left leg", "left_leg_raise"),
        ("raise your right leg", "right_leg_raise"),
        ("raise your leg", "right_leg_raise"),
        ("lift your leg", "right_leg_raise"),
        ("put your right leg down", "home"),
        ("put your left leg down", "home"),
        ("put your legs down", "home"),
        ("put your leg down", "home"),
        ("put the leg down", "home"),
        ("lower your right leg", "home"),
        ("lower your left leg", "home"),
        ("lower your legs", "home"),
        ("lower your leg", "home"),
        ("leg down", "home"),
        ("feet down", "home"),
        ("foot down", "home"),
        ("left leg out", "left_leg_out"),
        ("right leg out", "right_leg_out"),
        ("kneel on both knees", "kneel_both"),
        ("kneel on both of your knees", "kneel_both"),
        ("kneel on your knees", "kneel_both"),
        ("get on both knees", "kneel_both"),
        ("down on both knees", "kneel_both"),
        ("both knees down", "kneel_both"),
        ("double kneel", "kneel_both"),
        ("kneel both", "kneel_both"),
        ("kneel on both", "kneel_both"),
        ("kneel on your right knee", "kneel_right"),
        ("kneel on your left knee", "kneel_left"),
        ("kneel on the right knee", "kneel_right"),
        ("kneel on the left knee", "kneel_left"),
        ("get on your right knee", "kneel_right"),
        ("get on your left knee", "kneel_left"),
        ("down on your right knee", "kneel_right"),
        ("down on your left knee", "kneel_left"),
        ("get down on your right knee", "kneel_right"),
        ("get down on your left knee", "kneel_left"),
        ("right knee down", "kneel_right"),
        ("left knee down", "kneel_left"),
        ("kneel right", "kneel_right"),
        ("kneel left", "kneel_left"),
        ("right kneel", "kneel_right"),
        ("left kneel", "kneel_left"),
        ("kneel", "kneel_right"),
        ("lean to the right", "lean_right"),
        ("lean to the left", "lean_left"),
        ("lean right", "lean_right"),
        ("lean left", "lean_left"),
        ("tilt your body right", "lean_right"),
        ("tilt your body left", "lean_left"),
        ("tilt body right", "lean_right"),
        ("tilt body left", "lean_left"),
        ("give me a wave", "wave"),
        ("wave at me", "wave"),
        ("wave hello", "wave"),
        ("say hello with your hand", "wave"),
        ("body right", "lean_right"),
        ("body left", "lean_left"),
        ("torso right", "lean_right"),
        ("torso left", "lean_left"),
        *[
            (name.replace("_", " "), name)
            for name in POSES
            if name not in {"home", "neutral", "relaxed", "attention", "ready", "sit", "bow", "shrug"}
        ],
        *[
            (name, name)
            for name in POSES
            if name not in {"home", "neutral", "relaxed", "attention", "ready", "sit", "bow", "shrug"}
        ],
    ],
    key=lambda item: len(item[0]),
    reverse=True,
)

_ARM = re.compile(r"\b(?:arm|arms|hand|hands|shoulder|shoulders)\b", re.IGNORECASE)
_ELBOW = re.compile(r"\b(?:elbow|elbows)\b", re.IGNORECASE)
_WRIST = re.compile(r"\b(?:wrist|wrists)\b", re.IGNORECASE)
_LEG = re.compile(r"\b(?:leg|legs|hip|hips|thigh|thighs)\b", re.IGNORECASE)
_KNEE = re.compile(r"\b(?:knee|knees)\b", re.IGNORECASE)
_ANKLE = re.compile(r"\b(?:ankle|ankles|foot|feet)\b", re.IGNORECASE)
_HEAD = re.compile(r"\b(?:head|neck|face|chin)\b", re.IGNORECASE)
_TORSO = re.compile(r"\b(?:body|torso|back|waist|spine)\b", re.IGNORECASE)


_PLURAL_LIMB = re.compile(
    r"\b(?:hands|arms|shoulders|elbows|wrists|legs|knees|ankles|feet|hips)\b",
    re.IGNORECASE,
)
_OTHER_LIMB = re.compile(
    r"\b(?:the\s+other(?:\s+(?:one|hand|arm|side))?|(?:your\s+)?other\s+(?:hand|arm))\b",
    re.IGNORECASE,
)
_IDIOM_OTHER_HAND = re.compile(r"\bon the other hand\b", re.IGNORECASE)
_PUT_IT_DOWN = re.compile(
    r"\b(?:put it down|lower it|drop it|lower your hand|hand down|"
    r"put (?:your |the )?(?:left |right )?(?:leg|foot|knee)s? down|"
    r"(?:leg|foot|knee)s? down)\b",
    re.IGNORECASE,
)
_NOW_SIDE = re.compile(
    r"^(?:now\s+(?:the\s+)?(left|right)(?:\s+(?:one|hand|arm))?|(?:the\s+)(left|right)(?:\s+(?:one|hand|arm))?|(left|right)\s+(?:one|hand|arm))$",
    re.IGNORECASE,
)


def _raised_leg_sides(state: dict[str, Any] | None) -> list[str]:
    joints = (state or {}).get("joints") if isinstance(state, dict) else {}
    if not isinstance(joints, dict):
        joints = {}
    out: list[str] = []
    try:
        if float(joints.get("right_hip") or 0) >= 25:
            out.append("right")
        if float(joints.get("left_hip") or 0) >= 25:
            out.append("left")
    except (TypeError, ValueError):
        pass
    pose = str((state or {}).get("pose") or "") if isinstance(state, dict) else ""
    if pose in {"right_leg_raise", "right_leg_out"} and "right" not in out:
        out.append("right")
    if pose in {"left_leg_raise", "left_leg_out"} and "left" not in out:
        out.append("left")
    return out


def _raised_sides(state: dict[str, Any] | None) -> list[str]:
    joints = (state or {}).get("joints") if isinstance(state, dict) else {}
    if not isinstance(joints, dict):
        joints = {}
    out: list[str] = []
    try:
        if float(joints.get("right_shoulder") or 0) >= 80:
            out.append("right")
        if float(joints.get("left_shoulder") or 0) >= 80:
            out.append("left")
    except (TypeError, ValueError):
        return out
    return out


def _last_raised_side(state: dict[str, Any] | None) -> str | None:
    hist = (state or {}).get("history") if isinstance(state, dict) else None
    if not isinstance(hist, list):
        return None
    for ev in reversed(hist):
        if not isinstance(ev, dict):
            continue
        act = str(ev.get("act") or "")
        if act == "raise-right":
            return "right"
        if act == "raise-left":
            return "left"
    return None


def _other_side(state: dict[str, Any] | None) -> str:
    raised = _raised_sides(state)
    if raised == ["right"]:
        return "left"
    if raised == ["left"]:
        return "right"
    last = _last_raised_side(state)
    if last == "right":
        return "left"
    if last == "left":
        return "right"
    return "left"


def _mirror_user_side(side: str) -> str:
    """Front view: user's left is the arm on the left of the screen = robot's right."""
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    return side


def _sides(text: str) -> list[str]:
    has_l = bool(re.search(r"\bleft\b", text))
    has_r = bool(re.search(r"\bright\b", text))
    if has_l and has_r:
        return ["left", "right"]
    if has_l:
        return [_mirror_user_side("left")]
    if has_r:
        return [_mirror_user_side("right")]
    if re.search(r"\b(?:both|all)\b", text) or _PLURAL_LIMB.search(text):
        return ["left", "right"]
    return ["right"]


_JOINT_DIRS: dict[str, dict[str, int]] = {
    "neck_pan": {"left": 1, "right": -1},
    "neck_tilt": {"up": 1, "down": -1},
    "left_shoulder": {"fwd": 1, "forward": 1, "up": 1, "back": -1, "down": -1},
    "right_shoulder": {"fwd": 1, "forward": 1, "up": 1, "back": -1, "down": -1},
    "left_shoulder_out": {"out": 1, "in": -1},
    "right_shoulder_out": {"out": 1, "in": -1},
    "left_elbow": {"flex": 1, "bend": 1, "straight": -1, "extend": -1},
    "right_elbow": {"flex": 1, "bend": 1, "straight": -1, "extend": -1},
    "left_wrist": {"up": 1, "in": 1, "down": -1, "out": -1},
    "right_wrist": {"up": 1, "in": 1, "down": -1, "out": -1},
    "left_hip": {"fwd": 1, "forward": 1, "up": 1, "back": -1, "down": -1},
    "right_hip": {"fwd": 1, "forward": 1, "up": 1, "back": -1, "down": -1},
    "left_hip_out": {"out": 1, "in": -1},
    "right_hip_out": {"out": 1, "in": -1},
    "left_knee": {"flex": 1, "bend": 1, "straight": -1, "extend": -1},
    "right_knee": {"flex": 1, "bend": 1, "straight": -1, "extend": -1},
    "left_ankle": {"up": 1, "flex": 1, "down": -1},
    "right_ankle": {"up": 1, "flex": 1, "down": -1},
    "upper_back_pitch": {"fwd": 1, "forward": 1, "down": 1, "back": -1, "up": -1},
    "lower_back_pitch": {"fwd": 1, "forward": 1, "down": 1, "back": -1, "up": -1},
    "lower_back_roll": {"right": 1, "left": -1},
}


def _dir_sign(joint: str, direction: str) -> int:
    table = _JOINT_DIRS.get(_canon(joint) or joint) or {}
    key = str(direction or "").strip().lower()
    if key in table:
        return int(table[key])
    if key in {"plus", "+", "inc"}:
        return 1
    if key in {"minus", "-", "dec"}:
        return -1
    return 0


def _degree(text: str) -> float | None:
    m = _NUM.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


_DELTA_FOLLOW = re.compile(
    r"^(?:(?:please|ok|okay|yeah|yes)?\s*)?(?:"
    r"a (?:little|bit|tiny bit) (?:more|higher|lower|less|further|farther)|"
    r"(?:just )?(?:a )?(?:little |bit )?(?:higher|lower)|"
    r"(?:raise|lift|lower)(?:\s+it)?(?:\s+a)?(?:\s+(?:little|bit))?|"
    r"bend(?:\s+it)?(?:\s+a)?(?:\s+(?:little|bit))?(?:\s+more)?|"
    r"straighten(?:\s+it)?(?:\s+a)?(?:\s+(?:little|bit))?|"
    r"the other way|other (?:way|direction)|opposite(?:\s+way)?|"
    r"not so (?:high|much|far)|too (?:high|much|far)|"
    r"keep going|go further|more"
    r")(?:\s+please)?[.!?]*$",
    re.IGNORECASE,
)

# Whole-turn follow-ups that mean "repeat that move" or "same arm, new direction".
_REPEAT = re.compile(
    r"^(?:(?:please|ok|okay|yeah|yes|sure)\s+)*"
    r"(?:do (?:it|that)(?:\s+again)?|once more|again|same(?: thing)?|try (?:it|that) again)"
    r"(?:\s+please)?[.!?]*$",
    re.IGNORECASE,
)
_KIND_FOLLOW = re.compile(
    r"^(?:(?:please|ok|okay|yeah|yes|and|now|then)\s+)*"
    r"(?:go\s+|put it\s+|move it\s+|reach\s+)?"
    r"(?:forward|forwards|ahead|in front|up|down|out|back)"
    r"(?:\s+please)?[.!?]*$",
    re.IGNORECASE,
)


def _live_joints(state: dict[str, Any] | None) -> dict[str, float]:
    st = state if isinstance(state, dict) else {}
    raw_live = st.get("live") if isinstance(st.get("live"), dict) and st.get("live") else None
    raw = raw_live if raw_live is not None else st.get("joints")
    if not isinstance(raw, dict):
        raw = {}
    fallback = st.get("joints") if isinstance(st.get("joints"), dict) else {}
    out: dict[str, float] = {}
    for name in JOINTS:
        src = raw if name in raw else fallback
        try:
            out[name] = float(src.get(name) if src.get(name) is not None else POSES["home"].get(name, 0))
        except (TypeError, ValueError):
            out[name] = float(POSES["home"].get(name, 0))
    return out


def _last_applied(state: dict[str, Any] | None) -> dict[str, float]:
    hist = (state or {}).get("history") if isinstance(state, dict) else None
    if not isinstance(hist, list):
        return {}
    for ev in reversed(hist):
        if not isinstance(ev, dict):
            continue
        app = ev.get("applied")
        if isinstance(app, dict) and app:
            out: dict[str, float] = {}
            for k, v in app.items():
                name = _canon(str(k))
                clamped = _clamp(name, v)
                if clamped is not None:
                    out[name] = clamped
            if out:
                return out
    return {}


def _pose_focus(pose: str) -> list[str]:
    cur = POSES.get(pose) or {}
    home = POSES["home"]
    ranked = sorted(
        ((abs(float(cur.get(k, 0) or 0) - float(home.get(k, 0) or 0)), k) for k in JOINTS),
        reverse=True,
    )
    return [k for mag, k in ranked if mag >= 8][:4]


def _focus_names(state: dict[str, Any] | None, prior: str | None = None) -> list[str]:
    app = _last_applied(state)
    if app:
        return list(app)
    if prior and not is_delta_followup(prior):
        cmd = infer_command(prior, state)
        if isinstance(cmd, dict) and isinstance(cmd.get("joints"), dict) and cmd["joints"]:
            return [k for k in cmd["joints"] if k in JOINTS]
        if isinstance(cmd, dict) and cmd.get("pose") in POSES:
            return _pose_focus(str(cmd["pose"]))
    raised = _raised_sides(state)
    if len(raised) == 1:
        return [f"{raised[0]}_shoulder", f"{raised[0]}_elbow"]
    pose = str((state or {}).get("pose") or "") if isinstance(state, dict) else ""
    if pose in POSES and pose not in {"home", "neutral", "attention"}:
        return _pose_focus(pose)
    return []


def is_delta_followup(text: str) -> bool:
    t = " ".join((text or "").lower().split())
    return bool(t and _DELTA_FOLLOW.match(t))


def _infer_delta(text: str, state: dict[str, Any] | None, prior: str | None) -> dict[str, Any] | None:
    t = " ".join((text or "").lower().split())
    if not is_delta_followup(t):
        return None
    names = _focus_names(state, prior)
    if not names:
        return None
    if re.search(r"\b(?:higher|up)\b", t):
        prefer = [n for n in names if "shoulder" in n or n in {"neck_tilt", "upper_back_pitch"}]
        if prefer:
            names = prefer
    elif re.search(r"\b(?:bend|flex)\b", t):
        prefer = [n for n in names if "elbow" in n or "knee" in n]
        if prefer:
            names = prefer
    elif re.search(r"\bstraighten\b", t):
        prefer = [n for n in names if "elbow" in n or "knee" in n]
        if prefer:
            names = prefer
    amount = 18.0
    if re.search(r"\b(?:tiny|slightly|just a)\b", t):
        amount = 10.0
    elif re.search(r"\b(?:much|way|lot)\b", t):
        amount = 28.0
    deg = _degree(t)
    if deg is not None:
        amount = abs(deg)
    less = bool(re.search(r"\b(?:less|lower|down|not so|too (?:high|much|far)|straighten)\b", t))
    other_way = bool(re.search(r"\b(?:other way|other direction|opposite)\b", t))
    live = _live_joints(state)
    last = _last_applied(state)
    home = POSES["home"]
    out: dict[str, float] = {}
    for name in names:
        cur = live.get(name, 0.0)
        rest = float(home.get(name, 0) or 0)
        if other_way:
            lo, hi = JOINTS[name]
            if lo >= 0:
                nxt = rest if cur > rest + 8 else min(hi * 0.7, cur + 40)
            else:
                nxt = 2 * rest - cur
        else:
            ref = last.get(name, cur) if last else cur
            going_up = ref >= rest
            step = amount if going_up else -amount
            nxt = cur + (-step if less else step)
        clamped = _clamp(name, nxt)
        if clamped is not None:
            out[name] = clamped
    if not out:
        return None
    return {"cmd": "joint", "joints": out}


def _last_arm_robot_side(state: dict[str, Any] | None, prior: str | None = None) -> str | None:
    """Anatomical side of the last arm move (after viewer-mirror)."""
    app = _last_applied(state)
    for side in ("right", "left"):
        if any(
            k.startswith(f"{side}_") and any(p in k for p in ("shoulder", "elbow", "wrist"))
            for k in app
        ):
            return side
    raised = _raised_sides(state)
    if len(raised) == 1:
        return raised[0]
    last = _last_raised_side(state)
    if last:
        return last
    if prior and not _REPEAT.fullmatch(" ".join(prior.lower().split())) and not _KIND_FOLLOW.fullmatch(" ".join(prior.lower().split())):
        cmd = infer_command(prior, state, allow_plan=False)
        joints = (cmd or {}).get("joints") if isinstance(cmd, dict) else {}
        if isinstance(joints, dict):
            for side in ("right", "left"):
                if any(k.startswith(f"{side}_") and any(p in k for p in ("shoulder", "elbow", "wrist")) for k in joints):
                    return side
    return None


def _history_user_source(state: dict[str, Any] | None) -> str | None:
    hist = (state or {}).get("history") if isinstance(state, dict) else None
    if not isinstance(hist, list):
        return None
    for ev in reversed(hist):
        if not isinstance(ev, dict):
            continue
        user = " ".join(str(ev.get("user") or "").split())
        if user and not _REPEAT.fullmatch(user.lower()) and not _KIND_FOLLOW.fullmatch(user.lower()):
            return user
    return None


def _infer_repeat(text: str, state: dict[str, Any] | None, prior: str | None) -> dict[str, Any] | None:
    t = " ".join((text or "").lower().split())
    if not _REPEAT.fullmatch(t):
        return None
    source = " ".join((prior or "").split()) or _history_user_source(state)
    if source and " ".join(source.lower().split()) != t:
        hit = infer_command(source, state, allow_plan=False)
        if hit and hit.get("cmd") != "plan":
            return hit
    app = _last_applied(state)
    if app:
        return {"cmd": "joint", "joints": app}
    pose = str((state or {}).get("pose") or "") if isinstance(state, dict) else ""
    if pose in POSES and pose not in {"home", "neutral", "custom"}:
        return {"cmd": "pose", "pose": pose}
    return None


def _infer_kind_followup(text: str, state: dict[str, Any] | None, prior: str | None) -> dict[str, Any] | None:
    t = " ".join((text or "").lower().split())
    if not _KIND_FOLLOW.fullmatch(t):
        return None
    side = _last_arm_robot_side(state, prior)
    if not side:
        return None
    if _FWD_RE.search(t):
        kind = "forward"
    elif re.search(r"\b(?:up|raise)\b", t):
        kind = "raise"
    elif re.search(r"\b(?:down|lower)\b", t):
        kind = "lower"
    elif re.search(r"\bout\b", t):
        kind = "out"
    elif re.search(r"\bback\b", t):
        kind = "lower"
    else:
        return None
    return _arm_cmd(side, kind, _degree(t))


def _arm_joints(side: str, kind: str, deg: float | None) -> dict[str, float]:
    sh, el = f"{side}_shoulder", f"{side}_elbow"
    out = f"{side}_shoulder_out"
    if kind == "raise":
        return {sh: deg if deg is not None else 145, out: 0, el: 20 if deg is None else 20}
    if kind == "forward":
        # Body Actions → Arms Forward (left_shoulder/right_shoulder 78, elbow 8)
        return {sh: deg if deg is not None else 78, out: 0, el: 8 if deg is None else 8}
    if kind == "lower":
        return {sh: 0 if deg is None else deg, out: 0, el: 5}
    if kind == "out":
        return {sh: 0, out: deg if deg is not None else 90, el: 0}
    if kind == "bend":
        return {el: deg if deg is not None else 90}
    if kind == "straight":
        return {el: 0 if deg is None else deg}
    return {sh: deg if deg is not None else 145, out: 0, el: 20}


def _arm_cmd(side: str, kind: str, deg: float | None = None) -> dict[str, Any]:
    sides = ["left", "right"] if side in {"both", "all"} else [side]
    joints: dict[str, float] = {}
    for s in sides:
        joints.update(_arm_joints(s, kind, deg))
    out: dict[str, Any] = {"cmd": "joint", "joints": joints, "arm_kind": kind}
    if len(sides) == 1:
        out["arm_side"] = sides[0]
    else:
        out["arm_side"] = "both"
    return out


def normalize_plan(body: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep at most four concrete steps. Nested plans flatten."""
    if not isinstance(body, dict):
        return None
    raw = body.get("steps") if isinstance(body.get("steps"), list) else []
    steps: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        inner = str(item.get("cmd") or "")
        if inner == "plan":
            nested = normalize_plan(item)
            if nested:
                steps.extend(nested["steps"])
            continue
        got = validate_command(item)
        if got and got.get("cmd") != "plan":
            steps.append(got)
        if len(steps) >= 4:
            break
    if not steps:
        return None
    why = str(body.get("why") or "").strip()[:160]
    return {"cmd": "plan", "steps": steps, "why": why}


def step_hold_s(step: dict[str, Any] | None) -> float:
    if not isinstance(step, dict):
        return 1.6
    pose = str(step.get("pose") or "")
    cmd = str(step.get("cmd") or "")
    if cmd in {"stop", "stop_demo", "neutral"}:
        return 1.2
    if pose == "wave" or cmd == "walk":
        return 3.2
    if pose in {"sit", "squat", "bow"}:
        return 2.0
    return 1.6


def advance_plan(state: dict[str, Any]) -> dict[str, Any] | None:
    plan = state.get("plan") if isinstance(state, dict) else None
    if not isinstance(plan, dict):
        return None
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    i = int(plan.get("i") or 0) + 1
    if i >= len(steps):
        state["plan"] = None
        return None
    plan["i"] = i
    nxt = steps[i]
    return dict(nxt) if isinstance(nxt, dict) else None


def plan_speech(cmd: dict[str, Any] | None) -> str:
    if not isinstance(cmd, dict):
        return ""
    why = str(cmd.get("why") or "").strip()
    if why:
        return why
    steps = cmd.get("steps") if isinstance(cmd.get("steps"), list) else []
    names: list[str] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        if s.get("pose"):
            pose = str(s["pose"])
            if pose.endswith("_leg_raise"):
                names.append("raise my leg")
            else:
                names.append(pose.replace("_", " "))
        elif s.get("cmd") == "walk":
            d = str(s.get("direction") or "").strip().lower()
            names.append(f"walk {d}" if d and d not in {"place", ""} else "walk")
        elif s.get("cmd") in {"stop", "stop_demo", "neutral"}:
            names.append("stop")
        elif s.get("cmd") in {"joint", "joints"}:
            joints = s.get("joints") if isinstance(s.get("joints"), dict) else {}
            try:
                hip = max(float(joints.get("right_hip") or 0), float(joints.get("left_hip") or 0))
            except (TypeError, ValueError):
                hip = 0.0
            names.append("raise my leg" if hip >= 25 else "adjust")
    if not names:
        return "I'll do that in a few steps."
    if len(names) == 1:
        return f"I'll {names[0]}."
    return "I'll " + ", then ".join(names[:-1]) + ", then " + names[-1] + "."


def align_motor_to_text(cmd: dict[str, Any] | None, text: str) -> dict[str, Any] | None:
    """Force mentioned side (as seen on screen) and cap 'forward' so it is not a raise."""
    if not isinstance(cmd, dict):
        return cmd
    t = " ".join((text or "").lower().split())
    if not t:
        return cmd
    out = dict(cmd)
    joints = dict(out.get("joints") or {})
    named = _canon(str(out.get("joint") or ""))
    if named in JOINTS and out.get("value") is not None:
        joints.setdefault(named, out.get("value"))
    has_l = bool(re.search(r"\bleft\b", t))
    has_r = bool(re.search(r"\bright\b", t))
    fwd = bool(_FWD_RE.search(t))
    if has_l ^ has_r:
        want = _mirror_user_side("left" if has_l else "right")
        other = "left" if want == "right" else "right"
        remapped: dict[str, Any] = {}
        for k, v in joints.items():
            if k.startswith(other + "_"):
                remapped[want + k[len(other) :]] = v
            elif k.startswith(want + "_") or not k.startswith(("left_", "right_")):
                remapped[k] = v
        joints = remapped
        if named.startswith(other + "_"):
            out["joint"] = want + named[len(other) :]
    if fwd and not re.search(r"\boverhead\b", t):
        for side in ("left", "right"):
            sh = f"{side}_shoulder"
            if sh in joints:
                try:
                    ang = float(joints[sh])
                except (TypeError, ValueError):
                    ang = 0.0
                if ang > 110:
                    joints[sh] = 78.0
                    joints[f"{side}_shoulder_out"] = 0.0
                    el = f"{side}_elbow"
                    try:
                        if float(joints.get(el) or 0) > 25:
                            joints[el] = 8.0
                    except (TypeError, ValueError):
                        joints[el] = 8.0
        out["arm_kind"] = "forward"
    if joints:
        out["joints"] = joints
    return out


def parse_limb_intent(text: str) -> dict[str, Any] | None:
    """Viewer-relative limb intent. Source of truth for verify/correct — not the LLM."""
    t = " ".join((text or "").lower().split())
    if not t or _NEG.search(t) or looks_like_body_query(t):
        return None
    if _ELBOW.search(t):
        limb = "elbow"
    elif _WRIST.search(t):
        limb = "wrist"
    elif _KNEE.search(t):
        limb = "knee"
    elif _ANKLE.search(t):
        limb = "ankle"
    elif _LEG.search(t):
        limb = "leg"
    elif _ARM.search(t):
        limb = "arm"
    else:
        return None
    has_l = bool(re.search(r"\bleft\b", t))
    has_r = bool(re.search(r"\bright\b", t))
    both = bool(has_l and has_r) or bool(re.search(r"\b(?:both|all)\b", t)) or bool(_PLURAL_LIMB.search(t))
    if both:
        viewer_side = "both"
    elif has_l:
        viewer_side = "left"
    elif has_r:
        viewer_side = "right"
    else:
        viewer_side = None
    fwd = bool(_FWD_RE.search(t)) and not re.search(r"\boverhead\b", t)
    raise_v = bool(_RAISE_RE.search(t) or (re.search(r"\bup\b", t) and not fwd and not re.search(r"\bstand up\b", t)))
    lower_v = bool(
        re.search(r"\b(?:lower|drop)\b", t)
        or re.search(r"\b(?:hands?|arms?|legs?|feet|foot|knees?)\s+down\b", t)
        or re.search(r"\bput\b.{0,24}\b(?:leg|foot|knee|hand|arm).{0,12}\bdown\b", t)
    )
    out_v = bool(re.search(r"\b(?:out|aside|t-?pose)\b", t) and not fwd)
    bend_v = bool(re.search(r"\b(?:bend|flex|fold)\b", t))
    straight_v = bool(re.search(r"\b(?:straighten|extend|unbend)\b", t))
    kind = None
    if limb in {"arm", "elbow"}:
        if limb == "elbow":
            kind = "straight" if straight_v or lower_v else "bend"
        elif fwd:
            kind = "forward"
        elif raise_v and not lower_v:
            kind = "raise"
        elif lower_v:
            kind = "lower"
        elif out_v:
            kind = "out"
        elif bend_v:
            kind = "bend"
        elif straight_v:
            kind = "straight"
    elif limb in {"leg", "knee"}:
        if limb == "knee":
            kind = "straight" if straight_v or lower_v else "bend"
        elif raise_v or bend_v:
            kind = "raise"
        elif lower_v or straight_v:
            kind = "lower"
    if not kind:
        return None
    return {"limb": limb, "kind": kind, "viewer_side": viewer_side, "degrees": _degree(t)}


def _expected_robot_sides(intent: dict[str, Any] | None) -> list[str]:
    vs = str((intent or {}).get("viewer_side") or "")
    if vs == "both":
        return ["left", "right"]
    if vs in {"left", "right"}:
        return [_mirror_user_side(vs)]
    return ["right"]


def command_from_intent(intent: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(intent, dict) or not intent.get("kind"):
        return None
    kind = str(intent["kind"])
    limb = str(intent.get("limb") or "arm")
    deg = intent.get("degrees")
    joints: dict[str, float] = {}
    if limb in {"arm", "elbow"}:
        arm_kind = kind if limb == "arm" else ("straight" if kind == "straight" else "bend")
        sides = _expected_robot_sides(intent)
        if limb == "arm" and arm_kind in {"forward", "raise", "out", "lower", "bend", "straight"}:
            return _arm_cmd(sides[0] if len(sides) == 1 else "both", arm_kind, deg if isinstance(deg, (int, float)) else None)
        for side in sides:
            joints.update(_arm_joints(side, arm_kind, deg if isinstance(deg, (int, float)) else None))
    elif limb in {"leg", "knee"}:
        for side in _expected_robot_sides(intent):
            if kind in {"raise", "bend"}:
                joints[f"{side}_hip"] = deg if isinstance(deg, (int, float)) else (45 if limb == "leg" else 35)
                joints[f"{side}_knee"] = deg if isinstance(deg, (int, float)) and limb == "knee" else 20 if limb == "leg" else 90
            else:
                joints[f"{side}_hip"] = 0
                joints[f"{side}_knee"] = 5
    if not joints:
        return None
    return {"cmd": "joint", "joints": joints}


def _cmd_joints(cmd: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(cmd, dict):
        return {}
    joints: dict[str, float] = {}
    raw = cmd.get("joints")
    if isinstance(raw, dict):
        for k, v in raw.items():
            name = _canon(str(k))
            clamped = _clamp(name, v)
            if clamped is not None:
                joints[name] = clamped
    named = _canon(str(cmd.get("joint") or cmd.get("name") or ""))
    if named in JOINTS:
        clamped = _clamp(named, cmd.get("value", cmd.get("degrees", cmd.get("angle"))))
        if clamped is not None:
            joints[named] = clamped
    return joints


def motor_mismatch(cmd: dict[str, Any] | None, text: str) -> list[str]:
    """How a command disagrees with the user's limb intent. Empty = match."""
    intent = parse_limb_intent(text)
    if not intent:
        return []
    if not isinstance(cmd, dict):
        return ["no_command"]
    c = str(cmd.get("cmd") or cmd.get("command") or "").strip().lower()
    pose = str(cmd.get("pose") or cmd.get("name") or "").strip().lower().replace(" ", "_")
    vs = intent.get("viewer_side")
    kind = intent["kind"]
    if c == "pose" and pose in POSES:
        if kind == "raise" and vs == "both" and pose == "hands_up":
            return []
        if kind == "forward" and vs == "both" and pose == "arms_forward":
            return []
        if kind == "out" and pose == "tpose":
            return []
        if kind == "lower" and pose in {"home", "neutral", "attention"}:
            return []
        if vs in {"left", "right"} and pose in {"hands_up", "tpose", "arms_forward"}:
            return ["wrong_both"]
        return []
    if c not in {"joint", "joints", "set_joint", "nudge"}:
        if intent["limb"] in {"arm", "elbow", "leg", "knee", "wrist"}:
            return ["not_joint"]
        return []
    joints = _cmd_joints(cmd)
    issues: list[str] = []
    want_sides = _expected_robot_sides(intent)
    if vs in {"left", "right"}:
        want = want_sides[0]
        other = "left" if want == "right" else "right"
        primary = [k for k in joints if k.startswith(want + "_")]
        wrong = [k for k in joints if k.startswith(other + "_")]
        if wrong and not primary:
            issues.append("wrong_side")
    if intent["limb"] != "arm":
        return issues
    if kind == "forward":
        for side in want_sides:
            if f"{side}_shoulder" in joints and float(joints[f"{side}_shoulder"]) > 110:
                issues.append("forward_as_raise")
            if f"{side}_shoulder_out" in joints and float(joints[f"{side}_shoulder_out"]) > 40:
                issues.append("forward_as_out")
        if not any(f"{s}_shoulder" in joints for s in want_sides) and any("shoulder" in k for k in joints):
            issues.append("wrong_side")
    elif kind == "raise":
        for side in want_sides:
            if f"{side}_shoulder" in joints and float(joints[f"{side}_shoulder"]) < 110:
                issues.append("raise_too_low")
    elif kind == "out":
        for side in want_sides:
            if f"{side}_shoulder_out" in joints and float(joints[f"{side}_shoulder_out"]) < 50:
                issues.append("out_too_low")
            elif f"{side}_shoulder" in joints and float(joints[f"{side}_shoulder"]) > 50:
                issues.append("out_as_raise")
    return issues


def ensure_motor_matches_text(
    cmd: dict[str, Any] | None,
    text: str,
    state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Align side/kind, then rewrite from rules or intent if the command still mismatches."""
    if not isinstance(cmd, dict):
        return cmd
    if str(cmd.get("cmd") or "") == "plan":
        return cmd
    aligned = align_motor_to_text(cmd, text)
    if not motor_mismatch(aligned, text):
        return aligned
    rule = infer_command(text, state, allow_plan=False)
    if rule:
        fixed = align_motor_to_text(rule, text)
        if not motor_mismatch(fixed, text):
            return fixed
    rebuilt = command_from_intent(parse_limb_intent(text))
    return rebuilt or aligned


def joints_mismatch_intent(live: dict[str, Any] | None, text: str) -> list[str]:
    """Post-apply check: live joint numbers vs the user's limb intent."""
    intent = parse_limb_intent(text)
    if not intent or intent["limb"] not in {"arm", "elbow", "leg", "knee"}:
        return []
    joints = live if isinstance(live, dict) else {}
    sides = _expected_robot_sides(intent)
    issues: list[str] = []
    kind = intent["kind"]
    if intent["limb"] == "arm":
        if kind == "forward":
            for side in sides:
                if not (60 <= _joint(joints, f"{side}_shoulder") <= 110) or _joint(joints, f"{side}_shoulder_out") > 45:
                    issues.append("forward")
        elif kind == "raise":
            for side in sides:
                if _joint(joints, f"{side}_shoulder") < 115:
                    issues.append("raise")
        elif kind == "out":
            for side in sides:
                if _joint(joints, f"{side}_shoulder_out") < 50:
                    issues.append("out")
        elif kind == "lower":
            for side in sides:
                if _joint(joints, f"{side}_shoulder") > 35 or _joint(joints, f"{side}_shoulder_out") > 25:
                    issues.append("lower")
        elif kind == "bend":
            for side in sides:
                if _joint(joints, f"{side}_elbow") < 40:
                    issues.append("bend")
    elif intent["limb"] == "elbow":
        for side in sides:
            el = _joint(joints, f"{side}_elbow")
            if kind == "bend" and el < 40:
                issues.append("bend")
            if kind == "straight" and el > 25:
                issues.append("straight")
    return issues


def confirm_move(
    state: dict[str, Any] | None,
    motor: dict[str, Any] | None = None,
    text: str = "",
    *,
    query: bool = False,
) -> str:
    """Say what the twin actually did, in the same left/right the user used (screen side)."""
    st = state if isinstance(state, dict) else {}
    if query:
        return describe_body(st)
    cmd = str((motor or {}).get("cmd") or "")
    if cmd == "plan":
        return plan_speech(motor) or "I'll do that in a few steps."
    pose = str(st.get("pose") or "home")
    motion = str(st.get("motion") or "idle")
    live = tracking_snapshot(st).get("live") if isinstance(tracking_snapshot(st).get("live"), dict) else {}
    if not live:
        live = st.get("live") if isinstance(st.get("live"), dict) else {}
    if not live:
        live = st.get("joints") if isinstance(st.get("joints"), dict) else {}
    try:
        pan = float(live.get("neck_pan") or 0)
    except (TypeError, ValueError):
        pan = 0.0
    looking = bool(re.search(r"\b(?:look|head|face|pan)\b", text or "", re.I))
    if looking and pan <= -12:
        return "Looking left."
    if looking and pan >= 12:
        return "Looking right."
    if cmd in {"walk", "start_walk"} or motion == "walking" or pose == "walk-cycle":
        direction = str((motor or {}).get("direction") or st.get("walk_direction") or "").strip().lower()
        if direction in {"left", "right", "back", "backward", "backwards", "forward", "forwards"}:
            word = "back" if direction.startswith("back") else ("forward" if direction.startswith("forward") else direction)
            return f"Walking {word}."
        heading = str(st.get("heading") or heading_from_walk(direction or "place"))
        if heading == "south" and direction in {"place", "south", ""}:
            return "Walking in place, facing you, south."
        return f"Walking {heading}."
    intent = parse_limb_intent(text) if text else None
    if intent and intent["limb"] in {"arm", "elbow"} and intent.get("kind") in {"raise", "lower", "forward", "out", "bend", "straight"}:
        vs = intent.get("viewer_side")
        if vs == "both":
            who = "both arms"
        elif vs in {"left", "right"}:
            who = f"the arm on your {vs}"
        else:
            who = "the arm on your left"
        kind = intent["kind"]
        titled = who[0].upper() + who[1:]
        if kind == "forward":
            return "Both arms are reaching forward." if vs == "both" else f"{titled} is reaching forward."
        if kind == "raise":
            if vs == "both":
                if _hand_is_raised(live, "left") and _hand_is_raised(live, "right"):
                    return "Both arms are up."
                return "I sent a raise, but my arms are not up yet."
            teela_side = "left" if vs == "right" else "right"
            if _hand_is_raised(live, teela_side):
                return f"{titled} is up."
            return "I sent a raise, but the arm is not up yet."
        if kind == "lower":
            teela_side = "left" if vs == "right" else "right"
            if vs == "both":
                return "Arms down."
            if not _hand_is_raised(live, teela_side) and _joint(live, f"{teela_side}_shoulder") < 25:
                return f"{titled} is down."
            return "I sent a lower, but the arm is not down yet."
        if kind == "out":
            return "Both arms are out to the side." if vs == "both" else f"{titled} is out to the side."
        if kind == "bend":
            return "Bending both elbows." if vs == "both" else f"Bending the elbow on your {vs or 'left'}."
        if kind == "straight":
            return "Straightening both arms." if vs == "both" else f"Straightening the arm on your {vs or 'left'}."
    left_wave = _looks_like_wave(live, "left")
    right_wave = _looks_like_wave(live, "right")
    if observed_waving(st) or (
        (motion == "waving" or pose == "wave" or (motor or {}).get("pose") == "wave")
        and (left_wave or right_wave)
    ):
        if left_wave and not right_wave:
            return "Waving with my left hand."
        return "Waving."
    if motion == "waving" or pose == "wave" or (motor or {}).get("pose") == "wave":
        return "I sent a wave, but my hand is not in the wave hold yet."
    try:
        pan = float(live.get("neck_pan") or 0)
    except (TypeError, ValueError):
        pan = 0.0
    if pan <= -12:
        return "Looking left."
    if pan >= 12:
        return "Looking right."
    if cmd in {"stop", "stop_demo"}:
        return "Stopped."
    if cmd == "reset":
        return "Back to rest."
    intent = parse_limb_intent(text) if text else None
    if intent and intent["limb"] in {"arm", "elbow"}:
        vs = intent.get("viewer_side")
        if vs == "both":
            who = "both arms"
        elif vs in {"left", "right"}:
            who = f"the arm on your {vs}"
        else:
            who = "the arm on your left"
        kind = intent["kind"]
        titled = who[0].upper() + who[1:]
        if kind == "forward":
            return "Both arms are reaching forward." if vs == "both" else f"{titled} is reaching forward."
        if kind == "raise":
            if vs == "both":
                if _hand_is_raised(live, "left") and _hand_is_raised(live, "right"):
                    return "Both arms are up."
                return "I sent a raise, but my arms are not up yet."
            teela_side = "left" if vs == "right" else "right"
            if _hand_is_raised(live, teela_side):
                return f"{titled} is up."
            return "I sent a raise, but the arm is not up yet."
        if kind == "lower":
            teela_side = "left" if vs == "right" else "right"
            if vs == "both":
                return "Arms down."
            if not _hand_is_raised(live, teela_side) and _joint(live, f"{teela_side}_shoulder") < 25:
                return f"{titled} is down."
            return "I sent a lower, but the arm is not down yet."
        if kind == "out":
            return "Both arms are out to the side." if vs == "both" else f"{titled} is out to the side."
        if kind == "bend":
            return "Bending both elbows." if vs == "both" else f"Bending the elbow on your {vs or 'left'}."
        if kind == "straight":
            return "Straightening both arms." if vs == "both" else f"Straightening the arm on your {vs or 'left'}."
    joints = (motor or {}).get("joints") if isinstance((motor or {}).get("joints"), dict) else {}
    try:
        rh = float(joints.get("right_shoulder") or 0)
        lh = float(joints.get("left_shoulder") or 0)
        rout = float(joints.get("right_shoulder_out") or 0)
        lout = float(joints.get("left_shoulder_out") or 0)
    except (TypeError, ValueError):
        rh, lh, rout, lout = 0.0, 0.0, 0.0, 0.0
    if rh >= 120 and lh < 50:
        return "The arm on your left is up."
    if lh >= 120 and rh < 50:
        return "The arm on your right is up."
    if 70 <= rh <= 110 and lh < 50 and rout < 40:
        return "The arm on your left is reaching forward."
    if 70 <= lh <= 110 and rh < 50 and lout < 40:
        return "The arm on your right is reaching forward."
    if pose in _POSE_SHORT:
        return _POSE_SHORT[pose]
    if query:
        return _stance_phrase(st, st.get("joints") if isinstance(st.get("joints"), dict) else {})
    return "Done."


_PLAN_SPLIT = re.compile(
    r"\s*(?:,?\s+and then|,?\s+then|after that|afterwards|,?\s+and|,)\s+",
    re.I,
)

_DIR_ONLY = {
    "right": "right",
    "left": "left",
    "back": "back",
    "backward": "back",
    "backwards": "back",
    "forward": "south",
    "forwards": "south",
    "east": "left",
    "west": "right",
    "north": "back",
    "south": "south",
    "in place": "place",
}


def _direction_only(part: str) -> str | None:
    t = " ".join((part or "").lower().split()).strip(" .,!")
    t = re.sub(r"^(?:please\s+)?(?:walk(?:ing)?\s+)?", "", t)
    t = re.sub(r"^(?:to the|to|the)\s+", "", t)
    return _DIR_ONLY.get(t)


def compose_plan(text: str, state: dict[str, Any] | None = None, prior: str | None = None) -> dict[str, Any] | None:
    """Multi-step or open requests. She chooses how; she does not move unprompted."""
    t = " ".join((text or "").lower().split())
    if not t:
        return None
    parts = [p.strip(" .,!") for p in _PLAN_SPLIT.split(t) if p.strip(" .,!")]
    if len(parts) >= 2:
        steps: list[dict[str, Any]] = []
        for part in parts:
            hit = infer_command(part, state, prior, allow_plan=False)
            if (not hit or hit.get("cmd") == "plan") and steps:
                prev = str(steps[-1].get("cmd") or "")
                direction = _direction_only(part)
                if direction and prev in {"walk", "start_walk", "stop", "stop_demo", "neutral"}:
                    hit = {"cmd": "walk", "direction": direction}
            if hit and hit.get("cmd") != "plan":
                steps.append(hit)
        if len(steps) >= 2:
            return {"cmd": "plan", "steps": steps[:4], "why": plan_speech({"steps": steps[:4]})}
    pose = str((state or {}).get("pose") or "home") if isinstance(state, dict) else "home"
    if re.search(r"\b(?:get|make yourself) comfortable\b|\bsettle in\b|\btake a rest\b", t):
        if pose == "sit":
            return {"cmd": "pose", "pose": "relaxed"}
        return {"cmd": "pose", "pose": "sit"}
    if re.search(r"\bsay (?:hi|hello)\b|\bgreet(?: me| them)?\b", t):
        if pose == "wave" or str((state or {}).get("motion") or "") == "waving":
            return {"cmd": "joint", "joints": {"neck_tilt": 12}}
        return {"cmd": "pose", "pose": "wave"}
    if re.search(r"\bstretch(?:\s+(?:a bit|out))?\b", t) and not re.search(r"\bstretch of\b", t):
        return {
            "cmd": "plan",
            "why": "I'll stretch my arms up, then come back to rest.",
            "steps": [{"cmd": "pose", "pose": "hands_up"}, {"cmd": "pose", "pose": "home"}],
        }
    if re.search(r"\bget ready\b", t):
        return {"cmd": "pose", "pose": "ready"}
    if re.search(
        r"\b(?:do|perform|show(?:\s+me)?|give(?:\s+me)?)\b.{0,32}\b(?:some\s+)?(?:body\s+)?(?:move(?:ment)?s?|motions?|gestures?)\b"
        r"|\bmove(?:\s+your)?\s+(?:body|around)\b"
        r"|\bshow me you can move\b"
        r"|\buse your body\b"
        r"|\b(?:can|could) you move(?:\s+(?:around|your body))?\s*$",
        t,
    ):
        return {"cmd": "demo"}
    return None


def infer_command(text: str, state: dict[str, Any] | None = None, prior: str | None = None, *, allow_plan: bool = True) -> dict[str, Any] | None:
    """Map a motor request onto a pose, motion, joint set, or a short plan."""
    t = " ".join((text or "").lower().split())
    if not t or _blocked_by_negation(t):
        return None
    if looks_like_body_query(t):
        return None
    correcting = bool(
        re.search(
            r"\b(?:looks? (?:wrong|off|weird|better|too)|too (?:high|low|stiff|bent|straight)|should look)\b",
            t,
        )
    )
    if _IDIOM_OTHER_HAND.search(t) and not re.search(r"\b(?:raise|lift|lower|wave|sit|bow)\b", t):
        return None
    if allow_plan:
        planned = compose_plan(t, state, prior)
        if planned:
            return planned
    spoken = re.search(
        r"\b(left|right)\b.{0,24}\b(?:arm|hand)\b.{0,24}"
        r"(?:forward|forwards|ahead|in front|reach(?:ing)?(?:\s+out)?\s+(?:forward|ahead|in front|toward me|towards me|to me))"
        r"|(?:forward|forwards|ahead|in front|reach(?:ing)?(?:\s+out)?\s+(?:forward|ahead|in front|toward me|towards me|to me))"
        r".{0,20}\b(left|right)\b.{0,12}\b(?:arm|hand)\b",
        t,
    )
    if spoken and _ARM.search(t) and not re.search(r"\boverhead\b", t):
        side = _mirror_user_side((spoken.group(1) or spoken.group(2) or "right").lower())
        return _arm_cmd(side, "forward", _degree(t))
    if re.search(
        r"\b(?:arm|hand)\b.{0,24}(?:forward|forwards|ahead|in front|"
        r"reach(?:ing)?(?:\s+out)?\s+(?:forward|ahead|in front|toward me|towards me|to me))",
        t,
    ) and not re.search(r"\b(?:left|right|both|arms|hands|overhead)\b", t):
        return _arm_cmd("right", "forward", _degree(t))
    repeated = _infer_repeat(t, state, prior)
    if repeated:
        return repeated
    redirected = _infer_kind_followup(t, state, prior)
    if redirected:
        return redirected
    delta = _infer_delta(t, state, prior)
    if delta:
        return delta

    if _OTHER_LIMB.search(t):
        kind = "lower" if re.search(r"\b(?:lower|drop|down)\b", t) else "raise"
        return _arm_cmd(_other_side(state), kind, _degree(t))
    side_only = _NOW_SIDE.fullmatch(t)
    if side_only:
        spoken_side = next((g.lower() for g in side_only.groups() if g), "right")
        return _arm_cmd(_mirror_user_side(spoken_side), "raise", None)
    if _PUT_IT_DOWN.search(t) and not _PLURAL_LIMB.search(t):
        pose = str((state or {}).get("pose") or "") if isinstance(state, dict) else ""
        if re.search(r"\b(?:leg|foot|knee)s?\b", t) or "leg" in pose:
            return {"cmd": "pose", "pose": "home"}
        raised = _raised_sides(state)
        side = raised[0] if len(raised) == 1 else (_last_raised_side(state) or "right")
        return _arm_cmd(side, "lower", None)

    if re.search(r"\bestop off\b|\brelease e-?stop\b", t):
        return {"cmd": "estop_off"}
    if re.search(r"\bestop\b", t):
        return {"cmd": "estop_on"}
    if re.search(r"\bmotors? off\b", t):
        return {"cmd": "motors_off"}
    if re.search(r"\bmotors? on\b", t):
        return {"cmd": "motors_on"}
    if _STOP_MOTION.search(t):
        return {"cmd": "stop"}
    if re.search(r"\bwalk(?:ing)?\b", t) and not re.search(
        r"walk(?:ing)?\s+(?:me\s+)?through|walkthrough|\b(?:stop|halt|quit|end|cancel)\b.{0,16}\bwalk",
        t,
    ):
        if re.search(r"\b(?:back(?:wards?)?|away(?:\s+from me)?|north)\b", t):
            return {"cmd": "walk", "direction": "back"}
        if re.search(r"\b(?:toward(?:s)? me|forward|forwards|south)\b", t):
            return {"cmd": "walk", "direction": "south"}
        if re.search(r"\beast\b", t) or re.search(r"\bleft\b", t):
            return {"cmd": "walk", "direction": "left"}
        if re.search(r"\bwest\b", t) or re.search(r"\bright\b", t):
            return {"cmd": "walk", "direction": "right"}
        if re.search(r"\bin place\b", t):
            return {"cmd": "walk", "direction": "place"}
        return {"cmd": "walk"}
    if re.fullmatch(r"(?:please\s+)?(?:pose\s+)?demo", t) or re.search(r"\b(?:pose|walk(?:ing)?)\s+demo\b", t):
        return {"cmd": "demo"}
    if re.fullmatch(r"(?:please\s+)?reset", t) or re.search(r"\breset\s+(?:the\s+)?(?:pose|body|robot|joints?)\b", t):
        return {"cmd": "reset"}

    if re.search(r"\b(?:look|turn|face|pan)\b.{0,20}\bleft\b|\bhead\b.{0,12}\bleft\b", t):
        return {"cmd": "joint", "joint": "neck_pan", "value": _degree(t) if _degree(t) is not None else 40}
    if re.search(r"\b(?:look|turn|face|pan)\b.{0,20}\bright\b|\bhead\b.{0,12}\bright\b", t):
        return {"cmd": "joint", "joint": "neck_pan", "value": -(_degree(t) if _degree(t) is not None else 40)}
    if re.search(r"\b(?:look|tilt|face)\b.{0,20}\bup\b|\bhead\b.{0,12}\bup\b|\bnod up\b", t):
        return {"cmd": "joint", "joint": "neck_tilt", "value": _degree(t) if _degree(t) is not None else 20}
    if re.search(r"\b(?:look|tilt|face)\b.{0,20}\bdown\b|\bhead\b.{0,12}\bdown\b|\bnod\b", t):
        return {"cmd": "joint", "joint": "neck_tilt", "value": -(_degree(t) if _degree(t) is not None else 15)}
    if re.search(
        r"\b(?:look|face|turn)\b.{0,16}\b(?:straight|center|ahead|forward|at me)\b|"
        r"\bhead\b.{0,12}\b(?:straight|center|forward)\b",
        t,
    ):
        return {"cmd": "joint", "joint": "neck_pan", "value": 0}
    if _HEAD.search(t) and re.search(r"\bleft\b", t):
        return {"cmd": "joint", "joint": "neck_pan", "value": _degree(t) if _degree(t) is not None else 40}
    if _HEAD.search(t) and re.search(r"\bright\b", t):
        return {"cmd": "joint", "joint": "neck_pan", "value": -(_degree(t) if _degree(t) is not None else 40)}

    if (_TORSO.search(t) or re.search(r"\b(?:lean|tilt)\b", t)) and not _HEAD.search(t):
        if re.search(r"\b(?:forward|ahead|front)\b", t):
            return {"cmd": "joint", "joint": "upper_back_pitch", "value": _degree(t) if _degree(t) is not None else 18}
        if re.search(r"\b(?:back(?:ward)?|behind)\b", t):
            return {"cmd": "joint", "joint": "upper_back_pitch", "value": -(_degree(t) if _degree(t) is not None else 12)}
        if re.search(r"\bright\b", t) and not _ARM.search(t) and not _LEG.search(t):
            return {"cmd": "pose", "pose": "lean_right"}
        if re.search(r"\bleft\b", t) and not _ARM.search(t) and not _LEG.search(t):
            return {"cmd": "pose", "pose": "lean_left"}

    if not correcting:
        for phrase, pose in _POSE_PHRASES:
            if re.search(rf"\b{re.escape(phrase)}\b", t):
                return {"cmd": "pose", "pose": pose}

    joints: dict[str, float] = {}
    deg = _degree(t)
    sides = _sides(t)
    raise_v = bool(_RAISE_RE.search(t) or re.search(r"\b(?:hands?|arms?)\s+up\b", t))
    lower_v = bool(
        re.search(r"\b(?:lower|drop)\b", t)
        or re.search(r"\b(?:hands?|arms?|legs?|feet|foot|knees?)\s+down\b", t)
        or re.search(r"\bput\b.{0,24}\b(?:leg|foot|knee|hand|arm).{0,12}\bdown\b", t)
    )
    fwd_v = bool(_FWD_RE.search(t))
    out_v = bool(re.search(r"\b(?:out|aside|t-?pose)\b", t) and not fwd_v)
    bend_v = bool(re.search(r"\b(?:bend|flex|fold)\b", t))
    straight_v = bool(re.search(r"\b(?:straighten|extend|unbend)\b", t))

    if _ELBOW.search(t):
        kind = "straight" if straight_v or lower_v else "bend"
        for side in sides:
            joints.update(_arm_joints(side, kind, deg))
    elif _WRIST.search(t):
        for side in sides:
            if raise_v or re.search(r"\b(?:in|up)\b", t):
                joints[f"{side}_wrist"] = deg if deg is not None else 30
            elif lower_v or re.search(r"\b(?:out|down)\b", t):
                joints[f"{side}_wrist"] = -deg if deg is not None else -20
            else:
                joints[f"{side}_wrist"] = deg if deg is not None else 20
    elif _KNEE.search(t):
        for side in sides:
            if straight_v or lower_v:
                joints[f"{side}_knee"] = 0 if deg is None else deg
                joints[f"{side}_hip"] = 0
            else:
                joints[f"{side}_knee"] = deg if deg is not None else 90
                joints[f"{side}_hip"] = 35
    elif _ANKLE.search(t):
        for side in sides:
            joints[f"{side}_ankle"] = deg if deg is not None else (12 if raise_v else -10)
    elif _LEG.search(t):
        if not re.search(r"\b(?:left|right|both)\b", t):
            raised = _raised_leg_sides(state)
            if raised:
                sides = raised
        for side in sides:
            if raise_v or bend_v:
                joints[f"{side}_hip"] = deg if deg is not None else 45
                joints[f"{side}_knee"] = 20
            elif lower_v or straight_v or re.search(r"\bdown\b", t):
                joints[f"{side}_hip"] = 0
                joints[f"{side}_knee"] = 5
            else:
                joints[f"{side}_hip"] = deg if deg is not None else 30
                joints[f"{side}_knee"] = 20
    elif _ARM.search(t):
        if fwd_v and not re.search(r"\boverhead\b", t):
            kind = "forward"
        elif raise_v:
            kind = "raise"
        elif lower_v:
            kind = "lower"
        elif out_v:
            kind = "out"
        elif bend_v:
            kind = "bend"
        elif straight_v:
            kind = "straight"
        else:
            return None
        for side in sides:
            joints.update(_arm_joints(side, kind, deg))
        return _arm_cmd(sides[0] if len(sides) == 1 else "both", kind, deg)

    named = re.findall(
        r"\b(left|right)?_?(neck_pan|neck_tilt|upper_back_pitch|lower_back_roll|"
        r"(?:left_|right_)?(?:shoulder|elbow|wrist|hip|knee|ankle))\b",
        t,
    )
    if named and (deg is not None or raise_v or lower_v or bend_v or straight_v or out_v):
        for side, part in named:
            joint = part if part in JOINTS else f"{side}_{part}" if side else ""
            joint = _canon(joint)
            if joint in JOINTS:
                joints[joint] = deg if deg is not None else (JOINTS[joint][1] * 0.7)

    if joints:
        return {"cmd": "joint", "joints": joints}
    return None


_MOTOR_HINT = re.compile(
    r"\b(?:"
    r"robot_(?:status|joint|pose|motion)|"
    r"hands_up|t-?pose|estop|"
    r"wave|squat|bow|shrug|"
    r"sit(?:\s+down)?|stand(?:\s+(?:up|straight|still))?|"
    r"(?:raise|lower|lift).{0,28}(?:arms?|hands?|leg|knee|foot|feet|head|chin|shoulder)|"
    r"(?:arms?|hands?).{0,12}(?:up|down)|"
    r"the\s+other(?:\s+(?:one|hand|arm|side))?|"
    r"lean.{0,16}(?:left|right|forward|back|ahead)|"
    r"tilt(?:\s+your)?\s+(?:body|torso|head)|"
    r"nod|"
    r"look\s+(?:left|right|up|down)|"
    r"turn\s+(?:your\s+)?(?:head|body|neck|left|right)|"
    r"(?:start\s+)?walk(?:ing)?(?:\s+demo)?|"
    r"stop(?:\s+(?:walk|moving|that))?|"
    r"halt|stand\s+still|"
    r"(?:can|could|would|will)\s+you\s+stop|"
    r"(?:bend|flex|straighten|stretch)\s+your|"
    r"shrug|crouch|kneel|slump|hunch|stoop|"
    r"point\s+(?:at|to|your)|"
    r"(?:left|right)\s+(?:arm|hand|leg|shoulder|elbow|knee|hip|foot|wrist)|"
    r"move\s+your|"
    r"get comfortable|settle in|make yourself comfortable|"
    r"say (?:hi|hello)|greet(?: me| them)?|"
    r"stretch(?:\s+(?:a bit|out|your))|"
    r"get ready|"
    r"(?:do|perform|show).{0,24}(?:move(?:ment)?s?|motions?|gestures?)|"
    r"move(?:\s+your)?\s+(?:body|around)|"
    r"teela\s+body|"
    r"motors?\s+(?:on|off)|"
    r"reset\s+(?:pose|body|robot|joints?)"
    r")",
    re.IGNORECASE,
)


_BODY_QUERY = re.compile(
    r"(?:"
    r"\b(?:what|where|how)\b.+\b(?:arms?|hands?|pose|body|legs?|doing|position|joints?|look)|"
    r"\bwhat are you doing\b|"
    r"\bwhat do you look like\b|"
    r"\bhow do (?:you|i) look\b|"
    r"\bhow do you feel\b|"
    r"\bhow are you feeling\b|"
    r"\bhow(?:'s| is) your body\b|"
    r"\bwhat pose\b|"
    r"\b(?:current|right now).{0,24}\b(?:pose|body|position)|"
    r"\bhow(?:'s| is| are) (?:it going|your (?:body|pose|arm|hand|leg)|you looking)|"
    r"\bare you (?:sitting|standing|waving|walking|bowing|squatting|kneeling|crouching|leaning)|"
    r"\b(?:are|is)\s+your\s+(?:arms?|hands?|body|pose|head|legs?|knees?)|"
    r"\b(?:tell me|describe)\b.+\b(?:pose|arms?|hands?|body|yourself)|"
    r"\bwhat are you doing with your\s+(?:arms?|hands?)"
    r")",
    re.IGNORECASE,
)


def looks_like_body_query(text: str) -> bool:
    """True when the user is asking about pose, not commanding a move.

    'Can you raise your hands?' is a request. 'Are your hands up?' is a question.
    """
    t = " ".join((text or "").lower().split())
    if not t or _NEG.search(t):
        return False
    return bool(_BODY_QUERY.search(t))


def _arm_phrase(shoulder: Any, elbow: Any, shoulder_out: Any = 0) -> str:
    try:
        sh = float(shoulder or 0)
    except (TypeError, ValueError):
        sh = 0.0
    try:
        el = float(elbow or 0)
    except (TypeError, ValueError):
        el = 0.0
    try:
        out = float(shoulder_out or 0)
    except (TypeError, ValueError):
        out = 0.0
    if sh >= 120:
        loc = "raised up"
    elif out >= 55 and sh < 50:
        loc = "out to the side"
    elif 70 <= sh <= 110 and out < 40:
        loc = "reaching forward in front of me"
    elif el >= 85 and 18 <= sh < 85:
        loc = "in front of my chest"
    elif out >= 40 or sh >= 70:
        loc = "out to the side"
    elif sh >= 25:
        loc = "a little away from my side"
    else:
        loc = "down by my side"
    if el >= 70:
        loc += ", with the elbow bent"
    elif el >= 30:
        loc += ", with the elbow a little bent"
    else:
        loc += ", fairly straight"
    return loc


def _joint(joints: dict[str, Any], name: str) -> float:
    try:
        return float(joints.get(name) or 0)
    except (TypeError, ValueError):
        return 0.0


_POSE_SPEECH = {
    "home": "standing, facing you, south, weight over my feet",
    "neutral": "standing neutrally, facing you, south, weight over my feet",
    "relaxed": "standing relaxed, facing you, weight over my feet",
    "attention": "standing at attention, facing you, weight over my feet",
    "ready": "in a ready stance, facing you, weight over my feet",
    "sit": "sitting, weight over my feet",
    "bow": "bowing with my feet planted — hips back, head down, center of mass over my midfoot",
    "lean_left": "leaning to my left, weight over my left foot",
    "lean_right": "leaning to my right, weight over my right foot",
    "hands_up": "standing with both hands raised, weight over my feet",
    "tpose": "in a T-pose, arms out to the sides, weight over my feet",
    "wave": "waving with my right hand in front of my chest — not raised overhead — wrist rocking left and right, weight over my feet",
    "squat": "squatting, weight over my feet",
    "shrug": "shrugging, weight over my feet",
    "walk-cycle": "walking, weight shifting over the stance foot",
    "arms_forward": "standing with both arms reaching forward, weight over my feet",
    "bend_forward": "bending forward from the waist, weight over my feet",
    "ground_support": "in a low crouch with my hands forward, weight over my feet",
    "left_leg_out": "standing with my left leg out to the side, weight over my right foot",
    "right_leg_out": "standing with my right leg out to the side, weight over my left foot",
    "left_leg_raise": "standing on my right leg with my left leg raised",
    "right_leg_raise": "standing on my left leg with my right leg raised",
    "kneel_left": "kneeling on my left knee, right foot planted",
    "kneel_right": "kneeling on my right knee, left foot planted",
    "kneel_both": "kneeling on both knees",
}

_STANDING_POSES = {
    "home", "neutral", "relaxed", "attention", "ready", "hands_up", "tpose",
    "wave", "shrug", "arms_forward", "lean_left", "lean_right", "bow",
    "bend_forward", "ground_support",
}

_FAMILY_SPEECH = {
    "kneel_left": "I'm kneeling on my left knee, right foot planted.",
    "kneel_right": "I'm kneeling on my right knee, left foot planted.",
    "kneel_both": "I'm kneeling on both knees.",
    "sit": "I'm sitting, weight over my feet.",
    "squat": "I'm squatting, weight over my feet.",
    "left_leg_raise": "I'm standing on my right leg with my left leg raised.",
    "right_leg_raise": "I'm standing on my left leg with my right leg raised.",
    "left_leg_out": "I'm standing with my left leg out to the side, weight over my right foot.",
    "right_leg_out": "I'm standing with my right leg out to the side, weight over my left foot.",
}


def _hand_is_raised(joints: dict[str, Any], side: str = "right") -> bool:
    """True only for an overhead / hands-up raise, not the chest-front wave hold."""
    return _joint(joints, f"{side}_shoulder") >= 120


def _looks_like_wave(joints: dict[str, Any], side: str = "right") -> bool:
    """Wave hold: hand in front of the chest, elbow bent, shoulder only slightly up."""
    side = "left" if str(side or "right").lower() == "left" else "right"
    sh = _joint(joints, f"{side}_shoulder")
    el = _joint(joints, f"{side}_elbow")
    out = _joint(joints, f"{side}_shoulder_out")
    return el >= 70 and 12 <= sh < 85 and out < 40 and not _hand_is_raised(joints, side)


def _wave_is_rocking(state: dict[str, Any] | None, joints: dict[str, Any], side: str = "right") -> bool:
    """Wrist/elbow actually moving — a frozen leftover hold is not a wave."""
    wr = abs(_joint(joints, f"{side}_wrist"))
    el = _joint(joints, f"{side}_elbow")
    if wr >= 6:
        return True
    hold = (state or {}).get("wave_hold") if isinstance((state or {}).get("wave_hold"), dict) else {}
    if hold:
        if abs(el - _joint(hold, f"{side}_elbow")) >= 8:
            return True
        if abs(wr - abs(_joint(hold, f"{side}_wrist"))) >= 6:
            return True
    return False


def observed_waving(state: dict[str, Any] | None) -> bool:
    """True only when the live twin is actually rocking a wave, not leftover pose labels."""
    st = state if isinstance(state, dict) else {}
    if st.get("waving") is False:
        return False
    snap = tracking_snapshot(st)
    joints = snap.get("live") if isinstance(snap.get("live"), dict) else {}
    if not joints:
        joints = st.get("joints") if isinstance(st.get("joints"), dict) else {}
    side = "left" if _looks_like_wave(joints, "left") and not _looks_like_wave(joints, "right") else "right"
    if not (_looks_like_wave(joints, side) or _hand_is_raised(joints, "right")):
        return False
    motion = str(st.get("motion") or "")
    if not (st.get("waving") is True or motion == "waving"):
        return False
    return _wave_is_rocking(st, joints, side)


def _gait_foot_phrase(joints: dict[str, Any]) -> str:
    lh = _joint(joints, "left_hip")
    rh = _joint(joints, "right_hip")
    if lh > rh + 10:
        return " My left leg is swinging forward."
    if rh > lh + 10:
        return " My right leg is swinging forward."
    return ""


def _stance_family(joints: dict[str, Any]) -> str:
    """What the live angles actually look like — not the last pose label."""
    lh = _joint(joints, "left_hip")
    rh = _joint(joints, "right_hip")
    lk = _joint(joints, "left_knee")
    rk = _joint(joints, "right_knee")
    lout = _joint(joints, "left_hip_out")
    rout = _joint(joints, "right_hip_out")
    if lk >= 85 and lh <= 20 and rh >= 50:
        return "kneel_left"
    if rk >= 85 and rh <= 20 and lh >= 50:
        return "kneel_right"
    if lk >= 85 and rk >= 85 and lh <= 25 and rh <= 25:
        return "kneel_both"
    if lh >= 50 and rh >= 50 and lk >= 70 and rk >= 70:
        return "sit"
    if lh >= 32 and rh >= 32 and lk >= 55 and rk >= 55:
        return "squat"
    if lk >= 70 and lh >= 55 and rk < 40:
        return "left_leg_raise"
    if rk >= 70 and rh >= 55 and lk < 40:
        return "right_leg_raise"
    if lout >= 12 and rout < 8:
        return "left_leg_out"
    if rout >= 12 and lout < 8:
        return "right_leg_out"
    return "stand"


def _stance_phrase(st: dict[str, Any], joints: dict[str, Any]) -> str:
    motion = str(st.get("motion") or "idle")
    pose = str(st.get("pose") or "")
    if motion == "walking" or pose == "walk-cycle":
        heading = str(st.get("heading") or heading_from_walk(st.get("walk_direction")))
        foot = _gait_foot_phrase(joints)
        if heading == "south" and str(st.get("walk_direction") or "place") in {"place", "south", ""}:
            return "I'm walking in place, facing you, south." + foot
        return f"I'm walking {heading}, facing {heading}." + foot
    if _looks_like_wave(joints, "left") and not _looks_like_wave(joints, "right"):
        if _wave_is_rocking(st, joints, "left"):
            return (
                "I'm waving with my left hand in front of my chest, rocking the wrist — "
                "not raised overhead."
            )
        return "My left hand is held in a wave in front of my chest, not rocking — not raised overhead."
    if _looks_like_wave(joints):
        if _wave_is_rocking(st, joints, "right"):
            return (
                "I'm waving with my right hand in front of my chest, rocking the wrist left and right — "
                "not raised overhead."
            )
        return "My right hand is held in a wave in front of my chest, not rocking — not raised overhead."
    if observed_waving(st) and _hand_is_raised(joints, "right"):
        return "I'm waving with my right hand raised up, rocking it."
    if pose == "wave" and _looks_like_wave(joints):
        return "My right hand is held in a wave in front of my chest, not rocking — not raised overhead."
    if motion == "demo":
        return "I'm cycling through poses."
    family = _stance_family(joints)
    if family in _FAMILY_SPEECH:
        return _FAMILY_SPEECH[family]
    if pose in _POSE_SPEECH and family == "stand" and pose in _STANDING_POSES:
        if pose == "wave" and not _looks_like_wave(joints):
            pass
        else:
            return f"I'm {_POSE_SPEECH[pose]}."
    pitch = _joint(joints, "upper_back_pitch")
    hip = max(_joint(joints, "left_hip"), _joint(joints, "right_hip"))
    neck = _joint(joints, "neck_tilt")
    if hip >= 28 and pitch >= 16 and neck <= -8:
        return "I'm bowing with my feet planted, bent at the hips with my head down."
    if pitch >= 14 and hip < 20:
        return "I'm leaning forward from the waist."
    if _joint(joints, "lower_back_roll") >= 8:
        return "I'm leaning to my right."
    if _joint(joints, "lower_back_roll") <= -8:
        return "I'm leaning to my left."
    return "I'm standing, facing you, weight over my feet."


def _gravity_phrase(st: dict[str, Any], joints: dict[str, Any]) -> str:
    motion = str(st.get("motion") or "")
    if motion == "walking":
        return "Gravity is holding me down; my weight is over the stance foot."
    info = sagittal_support(joints)
    if not info.get("over_support"):
        if info["com_x"] < info["heel_x"]:
            return "My center of mass is behind my heels — I would fall backward."
        return "My center of mass is in front of my toes — I would fall forward."
    roll = _joint(joints, "lower_back_roll")
    if roll <= -8:
        return "Gravity keeps my weight over my left foot."
    if roll >= 8:
        return "Gravity keeps my weight over my right foot."
    return "Gravity keeps my weight over my feet."


def describe_body(
    state: dict[str, Any] | None,
    *,
    want_degrees: bool = False,
    now: float | None = None,
    which: str = "live",
) -> str:
    """Everyday description of the MiniOS twin. Degrees only if asked.

    which='live' is the 3D mesh / encoder snapshot. which='commanded' is the
    last pose the motors were told to hold.
    """
    st = state if isinstance(state, dict) else {}
    snap = tracking_snapshot(st, now=now)
    joints = snap["commanded"] if which == "commanded" else snap["live"]
    right = _arm_phrase(joints.get("right_shoulder"), joints.get("right_elbow"), joints.get("right_shoulder_out"))
    left = _arm_phrase(joints.get("left_shoulder"), joints.get("left_elbow"), joints.get("left_shoulder_out"))
    bits = [
        _stance_phrase(st, joints),
        f"My right arm is {right} (the arm on your left).",
        f"My left arm is {left} (the arm on your right).",
    ]
    bits.append(_gravity_phrase(st, joints))
    last = _last_applied(st)
    if last:
        if any(k.startswith("right_") for k in last) and not any(k.startswith("left_") for k in last):
            bits.append("The last thing I moved was my right arm (the arm on your left).")
        elif any(k.startswith("left_") for k in last) and not any(k.startswith("right_") for k in last):
            bits.append("The last thing I moved was my left arm (the arm on your right).")
        elif any("neck" in k for k in last):
            bits.append("The last thing I moved was my head.")
    pan = _joint(joints, "neck_pan")
    # Live HTML skeleton: negative pan is look left, positive is look right.
    # Match head_speech (±12) so I-feel and spoken look-lines agree.
    if pan <= -12:
        bits.append(f"My head is turned to my left ({int(round(pan))}° pan).")
    elif pan >= 12:
        bits.append(f"My head is turned to my right ({int(round(pan))}° pan).")
    else:
        bits.append(f"My head is facing straight ahead ({int(round(pan))}° pan).")
    if snap["phase"] is not None and not want_degrees:
        pct = int(round(float(snap["phase"]) * 100.0)) % 100
        bits.append(f"About {pct} percent through a walking step.")
    if want_degrees:
        def _deg(name: str) -> str:
            try:
                return str(int(round(float(joints.get(name) or 0))))
            except (TypeError, ValueError):
                return "0"
        bits.append(
            "Degrees — right shoulder "
            + _deg("right_shoulder")
            + ", right elbow "
            + _deg("right_elbow")
            + ", left shoulder "
            + _deg("left_shoulder")
            + ", left elbow "
            + _deg("left_elbow")
            + "."
        )
    if which != "commanded":
        skip_lag = str(st.get("motion") or "") in {"walking", "waving", "demo"}
        track = _tracking_speech(snap, want_degrees=want_degrees, skip_lag=skip_lag)
        if track:
            bits.append(track)
    return " ".join(bits)


_PAST_ACT = {
    "walk": "I was walking",
    "walk-cycle": "I was walking",
    "stop": "I stopped",
    "bow": "I bowed",
    "wave": "I waved",
    "sit": "I sat down",
    "home": "I stood at rest",
    "neutral": "I stood neutrally",
    "relaxed": "I stood relaxed",
    "attention": "I stood at attention",
    "ready": "I was in a ready stance",
    "hands_up": "I had both hands up",
    "tpose": "I was in a T-pose",
    "squat": "I was squatting",
    "shrug": "I shrugged",
    "lean_left": "I was leaning left",
    "lean_right": "I was leaning right",
    "demo": "I was running a pose demo",
    "reset": "I reset to rest",
    "raise-right": "I had my right hand up",
    "raise-left": "I had my left hand up",
    "forward-right": "I had my right arm reaching forward",
    "forward-left": "I had my left arm reaching forward",
    "out-right": "I had my right arm out to the side",
    "out-left": "I had my left arm out to the side",
    "lower": "I lowered my arms",
    "kneel_left": "I was kneeling on my left knee",
    "kneel_right": "I was kneeling on my right knee",
    "kneel_both": "I was kneeling on both knees",
    "arms_forward": "I had both arms reaching forward",
}


def relative_ago(ts: Any, now: float | None = None) -> str:
    now = time.time() if now is None else now
    try:
        dt = max(0.0, float(now) - float(ts))
    except (TypeError, ValueError):
        return "earlier"
    if dt < 10:
        return "just now"
    if dt < 60:
        n = int(dt)
        return f"{n} second{'s' if n != 1 else ''} ago"
    if dt < 3600:
        n = max(1, int(dt // 60))
        return f"{n} minute{'s' if n != 1 else ''} ago"
    n = max(1, int(dt // 3600))
    return f"{n} hour{'s' if n != 1 else ''} ago"


def act_from_command(body: dict[str, Any] | None, state: dict[str, Any] | None) -> str:
    body = body if isinstance(body, dict) else {}
    st = state if isinstance(state, dict) else {}
    cmd = str(body.get("cmd") or body.get("command") or "").strip().lower()
    if cmd in {"walk", "start_walk"}:
        return "walk"
    if cmd in {"stop", "stop_demo"}:
        return "stop"
    if cmd == "demo":
        return "demo"
    if cmd == "reset":
        return "reset"
    if cmd == "pose":
        return str(body.get("pose") or body.get("name") or st.get("pose") or "pose")
    if cmd in {"joint", "joints", "set_joint"}:
        joints = body.get("joints") if isinstance(body.get("joints"), dict) else {}
        name = str(body.get("joint") or "")

        def _arm_act(side: str) -> str:
            try:
                ang = float(joints.get(f"{side}_shoulder", body.get("value") or 0))
            except (TypeError, ValueError):
                ang = 0.0
            try:
                out = float(joints.get(f"{side}_shoulder_out") or 0)
            except (TypeError, ValueError):
                out = 0.0
            if ang >= 120:
                return f"raise-{side}"
            if 70 <= ang <= 110 and out < 40:
                return f"forward-{side}"
            if out >= 50:
                return f"out-{side}"
            if ang >= 90:
                return f"raise-{side}"
            return "lower"

        if "right_shoulder" in joints or name == "right_shoulder":
            return _arm_act("right")
        if "left_shoulder" in joints or name == "left_shoulder":
            return _arm_act("left")
        return "move"
    return str(st.get("pose") or st.get("motion") or "move")


def future_clause(cmd: dict[str, Any] | None) -> str:
    c = str((cmd or {}).get("cmd") or "")
    if c in {"walk", "start_walk"}:
        return "I will walk next."
    if c in {"stop", "stop_demo"}:
        return "I will stop and stand still next."
    if c == "demo":
        return "I will run a pose demo next."
    if c == "reset":
        return "I will return to rest next."
    if c == "pose":
        pose = str((cmd or {}).get("pose") or "").replace("_", " ")
        return f"I will {pose} next." if pose else "I will change pose next."
    if c in {"joint", "joints"}:
        return "I will move that part of my body next."
    return "I will follow that next."


def future_hold(state: dict[str, Any] | None) -> str:
    st = state if isinstance(state, dict) else {}
    plan = st.get("plan") if isinstance(st.get("plan"), dict) else None
    if plan:
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
        i = int(plan.get("i") or 0)
        rest = [s for s in steps[i + 1 :] if isinstance(s, dict)]
        if rest:
            speech = plan_speech({"steps": rest})
            if speech:
                return speech.replace("I'll", "Next I will", 1)
    motion = str(st.get("motion") or "")
    pose = str(st.get("pose") or "")
    if motion == "walking" or pose == "walk-cycle":
        return "I will keep walking until you tell me to stop."
    if motion == "demo":
        return "I will continue the pose demo until it ends or you stop me."
    if pose == "bow":
        return "I will stay in this bow until you ask me to stand or move."
    if pose == "sit":
        return "I will stay seated until you ask me to stand."
    if motion == "waving" or pose == "wave":
        return "I will keep waving until you ask me to stop or lower my hand."
    if pose == "hands_up":
        return "I will keep my hands up until you ask me to lower them."
    if pose == "tpose":
        return "I will hold the T-pose until you ask me to move."
    return "I will hold this until you ask me to move."


def remember_move(
    state: dict[str, Any],
    body: dict[str, Any] | None,
    *,
    user: str = "",
    now: float | None = None,
) -> None:
    """Append the current twin snapshot to a short timed history."""
    now = time.time() if now is None else now
    hist = [e for e in (state.get("history") or []) if isinstance(e, dict)]
    act = act_from_command(body, state)
    applied: dict[str, float] = {}
    raw = (body or {}).get("joints")
    if isinstance(raw, dict):
        for k, v in raw.items():
            name = _canon(str(k))
            clamped = _clamp(name, v)
            if clamped is not None:
                applied[name] = clamped
    joint = _canon(str((body or {}).get("joint") or (body or {}).get("name") or ""))
    if joint in JOINTS:
        clamped = _clamp(joint, (body or {}).get("value", (body or {}).get("degrees", (body or {}).get("angle"))))
        if clamped is not None:
            applied[joint] = clamped
    event = {
        "t": now,
        "act": act,
        "pose": state.get("pose"),
        "motion": state.get("motion"),
        "spoken": describe_body(state, which="commanded"),
        "user": " ".join((user or "").split())[:160],
        "applied": applied,
    }
    last = hist[-1] if hist else None
    if (
        last
        and last.get("act") == event["act"]
        and last.get("pose") == event["pose"]
        and last.get("motion") == event["motion"]
    ):
        last["t"] = now
        last["spoken"] = event["spoken"]
        if applied:
            last["applied"] = applied
    else:
        hist.append(event)
    state["history"] = hist[-12:]


def describe_body_timed(
    state: dict[str, Any] | None,
    *,
    upcoming: str | None = None,
    now: float | None = None,
    which: str = "live",
) -> str:
    """Past / present / future body speech for chat grounding."""
    now = time.time() if now is None else now
    st = state if isinstance(state, dict) else {}
    present = describe_body(st, now=now, which=which)
    hist = [e for e in (st.get("history") or []) if isinstance(e, dict)]
    past_bits: list[str] = []
    prior = hist[:-1] if hist else []
    for ev in reversed(prior[-3:]):
        ago = relative_ago(ev.get("t"), now)
        act = str(ev.get("act") or ev.get("pose") or "")
        clause = _PAST_ACT.get(act)
        if not clause:
            spoken = str(ev.get("spoken") or "").split(".")[0].strip()
            clause = spoken.replace("I'm ", "I was ") if spoken else ""
        if not clause:
            continue
        lead = clause[0].lower() + clause[1:] if clause else clause
        past_bits.append(f"{ago} {lead}")
    if past_bits:
        past = ". ".join(b[0].upper() + b[1:] if b else b for b in past_bits)
        if not past.endswith("."):
            past += "."
    else:
        past = "Before this I was at rest."
    if upcoming:
        future = upcoming.strip()
        if not future.endswith("."):
            future += "."
    else:
        future = future_hold(st)
    return f"Past: {past} Present: {present} Future: {future}"


_POSE_SHORT = {
    "home": "Standing straight.",
    "neutral": "Standing.",
    "relaxed": "Standing relaxed.",
    "attention": "Standing at attention.",
    "ready": "Ready.",
    "sit": "Sitting.",
    "bow": "Bowing.",
    "lean_left": "Leaning left.",
    "lean_right": "Leaning right.",
    "hands_up": "Hands up.",
    "tpose": "T-pose.",
    "wave": "Waving.",
    "waving": "Waving.",
    "squat": "Squatting.",
    "shrug": "Shrugging.",
    "walk-cycle": "Walking.",
    "arms_forward": "Arms forward.",
    "bend_forward": "Bending forward.",
    "ground_support": "Hands on the ground.",
    "left_leg_out": "Left leg out.",
    "right_leg_out": "Right leg out.",
    "left_leg_raise": "Left leg raised.",
    "right_leg_raise": "Right leg raised.",
    "kneel_right": "Kneeling on the right knee.",
    "kneel_left": "Kneeling on the left knee.",
    "kneel_both": "Kneeling on both knees.",
}


def short_reply(state: dict[str, Any] | None, motor: dict[str, Any] | None = None, *, query: bool = False) -> str:
    """One-line reply from the live twin — no LLM wait."""
    return confirm_move(state, motor, "", query=query)


_CHAT_GREET = re.compile(r"^(?:hi|hello|hey|yo|howdy)\b", re.I)
_CHAT_MEET = re.compile(r"\bnice to meet you\b|\bgood (?:morning|afternoon|evening)\b", re.I)
_CHAT_THANKS = re.compile(r"\bthanks?\b|\bthank you\b", re.I)
_CHAT_BYE = re.compile(r"\b(?:bye|goodbye|see you|cya)\b", re.I)
_CHAT_NOD = re.compile(
    r"^(?:(?:yeah|yep|yup|ok|okay|alright|got it|i see|makes sense|understood|mmhmm)[.!]*)+$",
    re.I,
)


def chat_acting(text: str, state: dict[str, Any] | None = None, prior: str | None = None) -> dict[str, Any] | None:
    """Small conversational body acting when chat is not an explicit motor command."""
    t = " ".join((text or "").lower().split())
    if not t or looks_like_body_query(t) or infer_command(t, state, prior):
        return None
    motion = str((state or {}).get("motion") or "") if isinstance(state, dict) else ""
    pose = str((state or {}).get("pose") or "") if isinstance(state, dict) else ""
    if motion in {"walking", "demo"}:
        return None
    if _CHAT_GREET.search(t) or _CHAT_MEET.search(t) or _CHAT_BYE.search(t):
        if motion == "waving" or pose == "wave":
            return None
        return {"cmd": "pose", "pose": "wave"}
    if _CHAT_THANKS.search(t):
        return {"cmd": "joint", "joint": "neck_tilt", "value": 12}
    if _CHAT_NOD.fullmatch(t) and motion != "waving":
        return {"cmd": "joint", "joint": "neck_tilt", "value": 10}
    return None


_GESTURE_SKIP = {
    "hello", "hi", "hey", "yo", "howdy", "thanks", "thank", "bye", "goodbye",
    "ok", "okay", "yes", "yeah", "yep", "please", "wave", "sit", "bow",
}


def command_from_named_gesture(text: str, gestures: dict[str, str] | None) -> dict[str, Any] | None:
    """User-taught names in BODY.md, e.g. 'my hello' → pose wave. Bare greetings stay chat."""
    if not gestures:
        return None
    t = " ".join((text or "").lower().split())
    if not t:
        return None
    for raw_name, raw_pose in sorted(gestures.items(), key=lambda kv: len(kv[0]), reverse=True):
        name = " ".join(str(raw_name).lower().split())
        pose = str(raw_pose or "").strip().lower().replace(" ", "_")
        if not name or not pose:
            continue
        if pose not in POSES and pose not in MOTIONS and pose not in {"walk", "stop", "demo", "reset"}:
            continue
        escaped = re.escape(name)
        cued = bool(
            re.search(rf"\b(?:my|your|the|that)\s+{escaped}\b", t)
            or re.search(rf"\b(?:do|show|perform)\s+(?:my\s+|the\s+|your\s+)?{escaped}\b", t)
            or re.search(rf"\b{escaped}\s+like before\b", t)
        )
        exact = t == name and name not in _GESTURE_SKIP
        if not cued and not exact:
            continue
        if pose in POSES:
            return {"cmd": "pose", "pose": pose}
        return {"cmd": pose if pose in MOTIONS else "walk"}
    return None


def looks_like_motor(text: str, state: dict[str, Any] | None = None, prior: str | None = None, gestures: dict[str, str] | None = None) -> bool:
    t = " ".join((text or "").lower().split())
    if not t or _blocked_by_negation(t):
        return False
    if looks_like_body_query(t):
        return False
    if _IDIOM_OTHER_HAND.search(t) and not re.search(r"\b(?:raise|lift|lower|wave|sit|bow)\b", t):
        return False
    t = re.sub(r"walk(?:ing)?\s+(?:me\s+)?through|walkthrough", " ", t)
    t = re.sub(r"\blook\s+at\b", " ", t)
    if command_from_named_gesture(t, gestures):
        return True
    if infer_command(t, state, prior) is not None:
        return True
    if re.search(
        r"\b(?:looks? (?:wrong|off|weird|better|too)|too (?:high|low|stiff|bent|straight)|should look)\b",
        t,
    ):
        return bool(is_delta_followup(t) and _focus_names(state, prior))
    if is_delta_followup(t):
        return bool(_focus_names(state, prior))
    if (_REPEAT.fullmatch(t) or _KIND_FOLLOW.fullmatch(t)) and (
        _last_arm_robot_side(state, prior) or _focus_names(state, prior) or prior
    ):
        return True
    return bool(_MOTOR_HINT.search(t))


def validate_command(body: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only commands robot_sim.apply() can run."""
    if not isinstance(body, dict):
        return None
    cmd = str(body.get("cmd") or body.get("command") or "").strip().lower()
    if cmd in {"", "null", "none", "status", "state"}:
        return None
    if cmd == "plan":
        return normalize_plan(body)
    if cmd == "pose":
        name = str(body.get("pose") or body.get("name") or "").strip().lower().replace(" ", "_")
        if name in POSES:
            return {"cmd": "pose", "pose": name}
        return None
    if cmd in {"joint", "joints", "set_joint", "nudge"}:
        updates: dict[str, float] = {}
        raw = body.get("joints")
        if isinstance(raw, dict):
            for k, v in raw.items():
                name = _canon(str(k))
                clamped = _clamp(name, v)
                if clamped is not None:
                    updates[name] = clamped
        joint = _canon(str(body.get("joint") or body.get("name") or ""))
        if joint in JOINTS:
            if body.get("delta") is not None or body.get("dir") or body.get("direction") or cmd == "nudge":
                out: dict[str, Any] = {"cmd": "joint", "joint": joint}
                if body.get("delta") is not None:
                    out["delta"] = body.get("delta")
                direction = str(body.get("dir") or body.get("direction") or "").strip()
                if direction:
                    out["dir"] = direction
                return out
            clamped = _clamp(joint, body.get("value", body.get("degrees", body.get("angle"))))
            if clamped is not None:
                updates[joint] = clamped
        if updates:
            out: dict[str, Any] = {"cmd": "joint", "joints": updates}
            kind = str(body.get("arm_kind") or "").strip().lower()
            if kind in {"forward", "raise", "out", "lower", "bend", "straight"}:
                out["arm_kind"] = kind
            side = str(body.get("arm_side") or "").strip().lower()
            if side in {"left", "right", "both"}:
                out["arm_side"] = side
            return out
        return None
    if cmd in MOTIONS or cmd in {"walk", "start_walk", "stop", "stop_demo", "neutral", "demo", "reset"}:
        out: dict[str, Any] = {"cmd": cmd}
        if cmd in {"walk", "start_walk"}:
            direction = str(body.get("direction") or "").strip().lower()
            if direction in {"left", "right", "place", "back", "south", "north", "east", "west", "forward", "backward", "backwards"}:
                out["direction"] = direction
        return out
    if cmd in {"walk_left", "walk_right", "walk_place", "walk_back", "walk_north", "walk_south", "walk_east", "walk_west"}:
        part = cmd.split("_", 1)[1]
        mapped = {"north": "back", "south": "south", "east": "left", "west": "right"}.get(part, part)
        return {"cmd": "walk", "direction": mapped}
    if cmd in {"estop_on", "estop_off", "motors_on", "motors_off"}:
        return {"cmd": cmd}
    return None


def heading_from_walk(direction: str | None) -> str:
    d = str(direction or "place").strip().lower()
    if d in {"left", "east"}:
        return "east"
    if d in {"right", "west"}:
        return "west"
    if d in {"back", "backward", "backwards", "north"}:
        return "north"
    return "south"


def default_state() -> dict[str, Any]:
    return {
        "joints": dict(POSES["home"]),
        "motors": True,
        "estop": False,
        "battery": 94,
        "pose": "home",
        "motion": "idle",
        "seq": 0,
        "walk_direction": "place",
        "heading": "south",
        "walk_started": None,
    }


def _canon(name: str) -> str:
    return ALIASES.get(str(name or "").strip(), str(name or "").strip())


def _clamp(name: str, value: Any) -> float | None:
    name = _canon(name)
    bounds = JOINTS.get(name)
    if not bounds:
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    lo, hi = bounds
    return max(lo, min(hi, n))


def _side_pt(origin: tuple[float, float], length: float, deg: float) -> tuple[float, float]:
    r = math.radians(deg)
    return (origin[0] + math.sin(r) * length, origin[1] + math.cos(r) * length)


def _rot_pelvis(p: tuple[float, float], pitch: float, pelvis: tuple[float, float] = (180.0, 282.0)) -> tuple[float, float]:
    r = math.radians(pitch)
    c, s = math.cos(r), math.sin(r)
    dx, dy = p[0] - pelvis[0], p[1] - pelvis[1]
    return (pelvis[0] + dx * c - dy * s, pelvis[1] + dx * s + dy * c)


def sagittal_support(joints: dict[str, Any] | None) -> dict[str, float]:
    """Side-view COM and support (heel/toe x) in the FK frame. +x is forward."""
    j = joints if isinstance(joints, dict) else {}
    pelvis = (180.0, 282.0)
    pitch = _joint(j, "upper_back_pitch")
    heels: list[float] = []
    toes: list[float] = []
    thigh_x: list[float] = []
    shin_x: list[float] = []
    for side, off in (("left", -6.0), ("right", 6.0)):
        hip = (180.0 + off, 307.0)
        ha = _joint(j, f"{side}_hip")
        ka = _joint(j, f"{side}_knee")
        aa = _joint(j, f"{side}_ankle")
        knee = _side_pt(hip, 88, ha)
        ankle = _side_pt(knee, 84, ha - ka)
        toe = _side_pt(ankle, 28, 90 + aa)
        heels.append(ankle[0])
        toes.append(toe[0])
        thigh_x.append((hip[0] + knee[0]) / 2)
        shin_x.append((knee[0] + ankle[0]) / 2)
    heel_x = sum(heels) / max(1, len(heels))
    toe_x = sum(toes) / max(1, len(toes))
    head = _rot_pelvis((181.0, 78.0), pitch, pelvis)
    torso = _rot_pelvis((180.0, 200.0), pitch, pelvis)
    arm_x: list[tuple[float, float]] = []
    for side, off in (("left", -6.0), ("right", 6.0)):
        sh0 = (180.0 + off, 157.0)
        sh_a = _joint(j, f"{side}_shoulder")
        el_a = _joint(j, f"{side}_elbow")
        e0 = _side_pt(sh0, 78, sh_a)
        w0 = _side_pt(e0, 74, sh_a - el_a)
        sh_r, e_r, w_r = _rot_pelvis(sh0, pitch, pelvis), _rot_pelvis(e0, pitch, pelvis), _rot_pelvis(w0, pitch, pelvis)
        arm_x.append(((sh_r[0] + e_r[0]) / 2, 0.03))
        arm_x.append(((e_r[0] + w_r[0]) / 2, 0.02))
    masses = [
        (head[0], 0.08),
        (torso[0], 0.36),
        (pelvis[0], 0.12),
        (sum(thigh_x) / 2, 0.20),
        (sum(shin_x) / 2, 0.09),
        *arm_x,
    ]
    tot = sum(m for _, m in masses) or 1.0
    com_x = sum(x * m for x, m in masses) / tot
    span = max(8.0, toe_x - heel_x)
    mid_x = heel_x + 0.40 * span
    if com_x < heel_x:
        err = com_x - heel_x
    elif com_x > toe_x:
        err = com_x - toe_x
    else:
        err = 0.0
    return {
        "com_x": com_x,
        "heel_x": heel_x,
        "toe_x": toe_x,
        "mid_x": mid_x,
        "err": err,
        "over_support": heel_x - 2.0 <= com_x <= toe_x + 2.0,
    }


def _set_joint(out: dict[str, Any], name: str, value: float) -> None:
    v = _clamp(name, value)
    if v is not None:
        out[name] = v


def apply_gravity(joints: dict[str, Any] | None, *, walking: bool = False) -> dict[str, Any]:
    """Every pose: planted feet, COM over the support so the body would not fall."""
    out = dict(joints or {})
    roll = _joint(out, "lower_back_roll")
    for side in ("left", "right"):
        hip = _joint(out, f"{side}_hip")
        knee = _joint(out, f"{side}_knee")
        ankle = _joint(out, f"{side}_ankle")
        if walking:
            if hip >= 4 and knee < 30:
                ankle = max(ankle, 2.0)
        elif hip > 8:
            knee = max(knee, min(hip * 0.55, 22.0))
            shin = hip - knee
            ankle = max(ankle, min(max(0.0, shin * 0.5), 12.0))
        # Coronal: put more weight on the loaded foot when leaning.
        if not walking and roll <= -6 and side == "left":
            knee = max(knee, 8.0)
            ankle = max(ankle, 4.0)
        if not walking and roll >= 6 and side == "right":
            knee = max(knee, 8.0)
            ankle = max(ankle, 4.0)
        ankle = max(ankle, -4.0)
        _set_joint(out, f"{side}_knee", knee)
        _set_joint(out, f"{side}_ankle", ankle)
    if walking:
        return out
    for _ in range(14):
        info = sagittal_support(out)
        err = info["err"]
        if abs(err) < 2.0:
            break
        adj = max(-2.8, min(2.8, err * 0.16))
        for side in ("left", "right"):
            _set_joint(out, f"{side}_hip", _joint(out, f"{side}_hip") + adj)
            hip = _joint(out, f"{side}_hip")
            _set_joint(out, f"{side}_knee", max(_joint(out, f"{side}_knee"), min(hip * 0.55, 22.0)))
    return out


def _fill_pose_joints() -> None:
    for pose in POSES.values():
        for name in JOINTS:
            pose.setdefault(name, 0.0)
        pose["right_wrist"] = 0.0 if pose is POSES.get("wave") else pose.get("right_wrist", 0.0)


def _bake_poses() -> None:
    _fill_pose_joints()
    for name, pose in list(POSES.items()):
        if name in {"kneel_left", "kneel_right", "kneel_both"}:
            continue
        POSES[name] = apply_gravity(dict(pose))
        if name == "wave":
            POSES[name]["right_wrist"] = 0.0


_bake_poses()

# Matches ui/robot-simulator.html applyWalkFrame: gaitPhase += dt * 5.1
WALK_PHASE_RAD_S = 5.1
_DELTA_SPEAK = 2.0


def _is_walking(state: dict[str, Any] | None) -> bool:
    st = state if isinstance(state, dict) else {}
    return str(st.get("motion") or "") == "walking" or str(st.get("pose") or "") == "walk-cycle"


def _clear_walk(state: dict[str, Any]) -> None:
    state["walk_started"] = None
    if "phase" in state:
        state["phase"] = None


def _full_joints(raw: Any, *, fallback: dict[str, float] | None = None) -> dict[str, float]:
    out = {name: 0.0 for name in JOINTS}
    if fallback:
        out.update(fallback)
    src = raw if isinstance(raw, dict) else {}
    for name in JOINTS:
        if name in src:
            v = _clamp(name, src[name])
            if v is not None:
                out[name] = v
    return out


def _reported(state: dict[str, Any], key: str) -> Any:
    if key not in state:
        return None
    return state[key]


def _cycle_frac(value: float) -> float:
    p = value % 1.0
    return 0.0 if p > 1.0 - 1e-9 else p


def walk_phase(state: dict[str, Any] | None, now: float | None = None) -> float | None:
    """Gait cycle 0..1 while walking; None when standing still."""
    st = state if isinstance(state, dict) else {}
    if not _is_walking(st):
        return None
    explicit = st.get("phase")
    if explicit is not None:
        try:
            p = float(explicit)
        except (TypeError, ValueError):
            p = None
        else:
            if math.isfinite(p):
                return _cycle_frac(p)
    started = st.get("walk_started")
    now_t = time.time() if now is None else float(now)
    try:
        t0 = float(started)
    except (TypeError, ValueError):
        return 0.0
    elapsed = max(0.0, now_t - t0)
    return _cycle_frac(elapsed * WALK_PHASE_RAD_S / (2.0 * math.pi))


def _ingest_live(state: dict[str, Any], raw: Any) -> None:
    """Record 3D-twin / Jetson angles onto state['live'] without moving commanded."""
    if not isinstance(raw, dict) or not raw:
        return
    cur = state.get("live") if isinstance(state.get("live"), dict) else {}
    merged = dict(cur)
    for raw_name, raw_val in raw.items():
        name = _canon(str(raw_name))
        clamped = _clamp(name, raw_val)
        if clamped is not None:
            merged[name] = clamped
    if merged:
        state["live"] = merged


def tracking_snapshot(state: dict[str, Any] | None, *, now: float | None = None) -> dict[str, Any]:
    """Commanded twin vs live 3D/encoder angles. Live equals commanded until something writes state['live']."""
    st = state if isinstance(state, dict) else {}
    commanded = _full_joints(st.get("joints"))
    raw_live = st.get("live")
    if isinstance(raw_live, dict) and raw_live:
        live = _full_joints(raw_live, fallback=commanded)
    else:
        live = dict(commanded)
    delta = {name: round(live[name] - commanded[name], 4) for name in JOINTS}
    return {
        "commanded": commanded,
        "live": live,
        "delta": delta,
        "phase": walk_phase(st, now),
        "temp": _reported(st, "temp"),
        "load": _reported(st, "load"),
        "fall_flag": _reported(st, "fall_flag"),
    }


def _tracking_speech(snap: dict[str, Any], *, want_degrees: bool, skip_lag: bool = False) -> str:
    delta = snap.get("delta") if isinstance(snap.get("delta"), dict) else {}
    lagging = [(n, d) for n, d in delta.items() if abs(float(d)) >= _DELTA_SPEAK]
    lagging.sort(key=lambda item: abs(item[1]), reverse=True)
    if want_degrees:
        bits: list[str] = []
        if skip_lag:
            pass
        elif not lagging:
            bits.append("Commanded matches live.")
        else:
            cmd = snap.get("commanded") if isinstance(snap.get("commanded"), dict) else {}
            live = snap.get("live") if isinstance(snap.get("live"), dict) else {}
            parts = []
            for name, d in lagging[:4]:
                parts.append(
                    f"{name.replace('_', ' ')} commanded {float(cmd.get(name) or 0):.0f} "
                    f"live {float(live.get(name) or 0):.0f} delta {d:+.0f}"
                )
            bits.append("Tracking — " + "; ".join(parts) + ".")
        if snap.get("phase") is not None:
            bits.append(f"Walk phase {float(snap['phase']):.2f}.")
        return " ".join(bits)
    if skip_lag or not lagging:
        return ""
    return "Part of my body is not matching the pose I am trying to hold."


def public_status(state: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    snap = tracking_snapshot(state, now=now)
    return {
        "ok": True,
        "joints": dict(snap["commanded"]),
        "commanded": snap["commanded"],
        "live": snap["live"],
        "delta": snap["delta"],
        "phase": snap["phase"],
        "temp": snap["temp"],
        "load": snap["load"],
        "fall_flag": snap["fall_flag"],
        "pose": state.get("pose") or "home",
        "motion": state.get("motion") or "idle",
        "waving": bool(state.get("waving")) if "waving" in state else (str(state.get("motion") or "") == "waving"),
        "motors": bool(state.get("motors", True)),
        "estop": bool(state.get("estop")),
        "battery": state.get("battery") or 94,
        "seq": int(state.get("seq") or 0),
        "joint_names": list(JOINTS),
        "poses": list(POSES),
        "wave_hold": dict(state.get("wave_hold") or {}) or None,
        "walk_direction": state.get("walk_direction") or "place",
        "heading": state.get("heading") or heading_from_walk(state.get("walk_direction")),
        "plan": dict(state.get("plan") or {}) or None,
        "spoken": describe_body(state, now=now),
        "hint": "Speak from spoken/live (the 3D twin). joints is commanded. Use robot_joint / robot_pose / robot_motion. Then desktop_watch to see MiniOS move.",
    }


def is_plan_echo(state: dict[str, Any] | None, body: dict[str, Any] | None) -> bool:
    """True when MiniOS is persisting the pose it just played for the current plan step."""
    if not isinstance(state, dict) or not isinstance(body, dict) or body.get("_plan_step"):
        return False
    cmd = str(body.get("cmd") or body.get("command") or "").strip().lower()
    if cmd in {"plan", "stop", "stop_demo", "neutral", "reset", "demo", "estop", "estop_on", "estop_off"}:
        return False
    plan = state.get("plan") if isinstance(state.get("plan"), dict) else None
    if not plan:
        return False
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    i = int(plan.get("i") or 0)
    if not (0 <= i < len(steps) and isinstance(steps[i], dict)):
        return False
    cur = steps[i]
    cur_cmd = str(cur.get("cmd") or "").strip().lower()
    pose = str(body.get("pose") or body.get("name") or "").strip().lower().replace(" ", "_")
    cur_pose = str(cur.get("pose") or cur.get("name") or "").strip().lower().replace(" ", "_")
    if cur_cmd in {"stop", "stop_demo", "neutral"}:
        if cmd in {"stop", "stop_demo", "neutral"}:
            return True
        if cmd in {"pose", "joints", "joint", "set_joint"} and pose in {
            "neutral", "home", "custom", "ready", "walk-cycle", "",
        }:
            return True
        if cmd in {"pose", "joints", "joint", "set_joint"} and pose == str(state.get("pose") or "").strip().lower():
            return True
    if cmd in {"pose", "joints", "joint", "set_joint"}:
        if cur_cmd == "pose" and pose and pose == cur_pose:
            return True
        if cur_cmd == "pose" and cur_pose == "wave" and pose in {"wave", "custom"}:
            return True
        if pose and pose == str(state.get("pose") or "").strip().lower():
            return True
    if cmd in {"walk", "start_walk"} and cur_cmd in {"walk", "start_walk"}:
        return True
    return False


def apply(state: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    cmd = str(body.get("cmd") or body.get("command") or "status").strip().lower()
    if cmd in {"walk_left", "walk_right", "walk_place", "walk_back", "walk_north", "walk_south", "walk_east", "walk_west"}:
        part = cmd.split("_", 1)[1]
        mapped = {"north": "back", "south": "south", "east": "left", "west": "right"}.get(part, part)
        body = dict(body)
        body["cmd"] = "walk"
        body["direction"] = mapped
        cmd = "walk"
    if isinstance(body.get("live"), dict):
        _ingest_live(state, body.get("live"))
    if cmd in {"live", "telemetry"}:
        # Observed 3D twin is authoritative for pose/motion — do not keep a
        # commanded "wave" label after the HTML body has stopped or never started.
        if body.get("pose") not in (None, ""):
            state["pose"] = str(body.get("pose")).strip().lower()
        if body.get("motion") not in (None, ""):
            state["motion"] = str(body.get("motion")).strip().lower()
        if "waving" in body:
            state["waving"] = bool(body.get("waving"))
        elif str(state.get("motion") or "") != "waving":
            state["waving"] = False
        raw_phase = body.get("phase")
        if raw_phase is not None and _is_walking(state):
            try:
                state["phase"] = _cycle_frac(float(raw_phase))
            except (TypeError, ValueError):
                pass
        elif not _is_walking(state):
            state["phase"] = None
        return public_status(state)
    if cmd in {"status", "state"}:
        return public_status(state)

    if cmd in {"estop_on", "estop_off"} or (cmd == "estop" and "on" in str(body.get("value") or body.get("estop") or "").lower()):
        on = cmd == "estop_on" or str(body.get("value") or body.get("estop") or "on").lower() in {"1", "true", "on", "yes"}
        if cmd == "estop_off":
            on = False
        state["estop"] = on
        state["motion"] = "stopped" if on else "idle"
        state["plan"] = None
        _clear_walk(state)
        state["seq"] = int(state.get("seq") or 0) + 1
        return public_status(state)

    if cmd in {"motors_on", "motors_off"} or cmd == "motors":
        on = cmd == "motors_on" or str(body.get("value") or body.get("motors") or "on").lower() in {"1", "true", "on", "yes"}
        if cmd == "motors_off":
            on = False
        state["motors"] = on
        if not on:
            state["motion"] = "idle"
            _clear_walk(state)
        state["seq"] = int(state.get("seq") or 0) + 1
        return public_status(state)

    if state.get("estop"):
        out = public_status(state)
        out.update({"ok": False, "error": "E-STOP is active"})
        return out
    if not state.get("motors", True) and cmd not in {"reset", "status"}:
        out = public_status(state)
        out.update({"ok": False, "error": "Motors are disabled"})
        return out

    if cmd not in {"plan"} and not body.get("_plan_step") and not is_plan_echo(state, body):
        state["plan"] = None

    if cmd == "plan":
        planned = normalize_plan(body)
        if not planned:
            out = public_status(state)
            out.update({"ok": False, "error": "Empty movement plan"})
            return out
        state["plan"] = {"steps": planned["steps"], "i": 0, "why": planned.get("why") or ""}
        first = dict(planned["steps"][0])
        first["_plan_step"] = True
        out = apply(state, first)
        out["why"] = planned.get("why") or ""
        out["plan"] = state.get("plan")
        out["plan_hold"] = step_hold_s(first)
        return out

    if cmd == "reset":
        state.update(default_state())
        state.pop("live", None)
        return public_status(state)

    if cmd in {"walk", "start_walk"}:
        if not _is_walking(state):
            state["walk_started"] = time.time()
            if "phase" in state:
                state["phase"] = None
        state["motion"] = "walking"
        state["pose"] = "walk-cycle"
        direction = str(body.get("direction") or "place").strip().lower()
        aliases = {
            "east": "left",
            "west": "right",
            "north": "back",
            "backward": "back",
            "backwards": "back",
            "forward": "south",
        }
        direction = aliases.get(direction, direction)
        if direction not in {"left", "right", "place", "back", "south"}:
            direction = "place"
        state["walk_direction"] = direction
        state["heading"] = heading_from_walk(direction)
        state["seq"] = int(state.get("seq") or 0) + 1
        return public_status(state)

    if cmd in {"stop", "stop_demo", "neutral"}:
        state["motion"] = "idle"
        _clear_walk(state)
        if not body.get("_plan_step"):
            state["plan"] = None
        if state.get("pose") == "walk-cycle":
            state["pose"] = "ready"
            state["joints"] = apply_gravity(dict(POSES["ready"]))
        else:
            # Body Actions Neutral: drop the gesture and rest.
            state["pose"] = "neutral"
            state["joints"] = apply_gravity(dict(POSES["neutral"]))
        state["live"] = dict(state.get("joints") or {})
        state["walk_direction"] = "place"
        state["heading"] = "south"
        state["seq"] = int(state.get("seq") or 0) + 1
        return public_status(state)

    if cmd == "demo":
        state["motion"] = "demo"
        state["pose"] = "attention"
        state["joints"] = apply_gravity(dict(POSES["attention"]))
        _clear_walk(state)
        state["seq"] = int(state.get("seq") or 0) + 1
        return public_status(state)

    if cmd == "pose":
        name = str(body.get("pose") or body.get("name") or "").strip().lower().replace(" ", "_")
        if name not in POSES:
            out = public_status(state)
            out.update({"ok": False, "error": f"Unknown pose {name!r}. Try: {', '.join(POSES)}"})
            return out
        hold = dict(POSES[name])
        extra = body.get("joints")
        if name == "wave":
            stored = state.get("wave_hold")
            # Keep a one-hand hold. Ignore the old both-elbows-135 dump.
            if isinstance(stored, dict) and stored and float(stored.get("left_elbow") or 0) < 40:
                hold.update({k: stored[k] for k in JOINTS if k in stored})
        if isinstance(extra, dict) and extra:
            for raw_name, raw_val in extra.items():
                jn = _canon(str(raw_name))
                clamped = _clamp(jn, raw_val)
                if clamped is not None:
                    hold[jn] = clamped
        if name in {"kneel_left", "kneel_right", "kneel_both"}:
            state["joints"] = dict(hold)
        else:
            state["joints"] = apply_gravity(hold)
        state["pose"] = name
        if name == "wave":
            mot = str(body.get("motion") or "waving").strip().lower()
            state["joints"]["right_wrist"] = 0.0
            # motion=idle means the overlay stopped (Arms Forward / Stop / persist after leaving).
            state["motion"] = "idle" if mot in {"idle", "stop", "stopped"} else "waving"
            state["wave_hold"] = {k: state["joints"].get(k) for k in JOINTS}
            state["wave_hold"]["right_wrist"] = 0.0
        else:
            state["motion"] = "idle"
        _clear_walk(state)
        state["seq"] = int(state.get("seq") or 0) + 1
        return public_status(state)

    if cmd in {"joint", "joints", "set_joint", "nudge"}:
        updates: dict[str, Any] = {}
        if isinstance(body.get("joints"), dict):
            updates.update(body["joints"])
        joint = _canon(str(body.get("joint") or body.get("name") or ""))
        direction = str(body.get("dir") or body.get("direction") or "").strip().lower()
        raw_delta = body.get("delta")
        raw_val = body.get("value", body.get("degrees", body.get("angle")))
        if joint:
            if raw_val is not None and raw_delta is None and not direction and cmd != "nudge":
                updates[joint] = raw_val
            else:
                cur = _joint(state.get("joints") if isinstance(state.get("joints"), dict) else {}, joint)
                sign = _dir_sign(joint, direction) if direction else 1
                if raw_delta is not None:
                    try:
                        updates[joint] = cur + sign * abs(float(raw_delta))
                    except (TypeError, ValueError):
                        updates[joint] = cur + sign * 8.0
                elif direction or cmd == "nudge":
                    if not sign and direction:
                        out = public_status(state)
                        out.update({"ok": False, "error": f"Unknown direction {direction!r} for {joint}"})
                        return out
                    updates[joint] = cur + (sign or 1) * 8.0
                elif raw_val is not None:
                    updates[joint] = raw_val
        if not updates:
            out = public_status(state)
            out.update({"ok": False, "error": "joint and value required"})
            return out
        applied = []
        skipped: list[str] = []
        for raw_name, raw_val in updates.items():
            name = _canon(str(raw_name))
            if name not in JOINTS:
                skipped.append(str(raw_name))
                continue
            v = _clamp(name, raw_val)
            if v is None:
                out = public_status(state)
                out.update({"ok": False, "error": f"Invalid angle for {name}"})
                return out
            state.setdefault("joints", {})[name] = v
            applied.append({"joint": name, "value": v})
        if not applied:
            out = public_status(state)
            unknown = skipped[0] if skipped else "joint"
            out.update({"ok": False, "error": f"Unknown joint {unknown!r}. Try: {', '.join(JOINTS)}"})
            return out
        posed = str(body.get("pose") or "").strip().lower().replace(" ", "_")
        mot = str(body.get("motion") or "").strip().lower()
        if posed == "wave" or mot == "waving":
            state["pose"] = "wave"
            state["motion"] = "waving" if mot == "waving" else "idle"
            state["wave_hold"] = {k: state["joints"].get(k) for k in JOINTS}
        else:
            state["pose"] = posed if posed in POSES else "custom"
            state["motion"] = "idle"
        _clear_walk(state)
        state["seq"] = int(state.get("seq") or 0) + 1
        out = public_status(state)
        out["set"] = applied
        return out

    out = public_status(state)
    out.update({"ok": False, "error": f"Unknown robot command {cmd!r}"})
    return out
