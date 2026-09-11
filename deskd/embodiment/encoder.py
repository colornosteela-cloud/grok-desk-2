#!/usr/bin/env python3
"""Compact body concepts for Qwen. Not 50 Hz telemetry."""
from __future__ import annotations

from typing import Any

import robot_sim
from embodiment.schema import clamp_joints


def _arm_state(joints: dict[str, float], side: str) -> dict[str, Any]:
    sh = float(joints.get(f"{side}_shoulder") or 0)
    el = float(joints.get(f"{side}_elbow") or 0)
    out = float(joints.get(f"{side}_shoulder_out") or 0)
    if robot_sim._looks_like_wave(joints) and side == "right":
        label = "wave_hold"
    elif sh >= 120:
        label = "raised"
    elif sh >= 60 and out < 40:
        label = "extended"
    elif out >= 50:
        label = "out"
    else:
        label = "down"
    return {
        "state": label,
        "shoulder_pitch_deg": round(sh, 1),
        "elbow_deg": round(el, 1),
        "shoulder_out_deg": round(out, 1),
    }


def _hand_state(joints: dict[str, float], side: str, held: str | None) -> dict[str, Any]:
    del joints
    if held:
        return {"state": "grasping", "object": held}
    return {"state": "open", "object": None}


def _support(joints: dict[str, float], pose: str) -> dict[str, Any]:
    pose = str(pose or "")
    if pose in {"kneel_both", "sit"}:
        support = "both_knees" if pose == "kneel_both" else "seat"
        return {"stable": True, "support": support}
    if pose == "kneel_left":
        return {"stable": True, "support": "left_knee_right_foot"}
    if pose == "kneel_right":
        return {"stable": True, "support": "right_knee_left_foot"}
    if pose in {"left_leg_raise"}:
        return {"stable": True, "support": "right_foot"}
    if pose in {"right_leg_raise"}:
        return {"stable": True, "support": "left_foot"}
    sag = robot_sim.sagittal_support(joints)
    err = abs(float(sag.get("err") or 0))
    return {"stable": err < 16, "support": "both_feet"}


def encode_body(observed: dict[str, Any], *, held: dict[str, str] | None = None) -> dict[str, Any]:
    """Semantic body block from the observed (authoritative) layer."""
    joints = clamp_joints(observed.get("joints"))
    pose = str(observed.get("pose") or "home")
    motion = str(observed.get("motion") or "idle")
    held = held if isinstance(held, dict) else {}
    health = observed.get("health") if isinstance(observed.get("health"), dict) else {}
    return {
        "balance": _support(joints, pose),
        "head": {
            "pan_deg": round(float(joints.get("neck_pan") or 0), 1),
            "tilt_deg": round(float(joints.get("neck_tilt") or 0), 1),
        },
        "right_arm": _arm_state(joints, "right"),
        "left_arm": _arm_state(joints, "left"),
        "right_hand": _hand_state(joints, "right", held.get("right_hand")),
        "left_hand": _hand_state(joints, "left", held.get("left_hand")),
        "contact": {
            "right_hand": bool(held.get("right_hand")),
            "left_foot": pose not in {"left_leg_raise"},
            "right_foot": pose not in {"right_leg_raise"},
        },
        "active_action": {
            "skill": None if motion in {"idle", ""} else motion,
            "state": "completed" if motion in {"idle", ""} else "active",
        },
        "health": {
            "joints_ok": not bool(health.get("emergency_stop")),
            "emergency_stop": bool(health.get("emergency_stop") or observed.get("estop")),
            "motors": bool(observed.get("motors", True)),
        },
    }


def prediction_errors(
    intended_joints: dict[str, float],
    observed_joints: dict[str, float],
    *,
    threshold_deg: float = 2.0,
) -> list[dict[str, Any]]:
    """First-class intended vs observed joint error. Empty when they match."""
    want = clamp_joints(intended_joints)
    got = clamp_joints(observed_joints)
    out: list[dict[str, Any]] = []
    for name in want:
        expected = float(want[name])
        actual = float(got[name])
        diff = actual - expected
        if abs(diff) < threshold_deg:
            continue
        out.append(
            {
                "joint": name,
                "expected_deg": round(expected, 1),
                "actual_deg": round(actual, 1),
                "difference_deg": round(diff, 1),
            }
        )
    out.sort(key=lambda row: abs(float(row["difference_deg"])), reverse=True)
    return out


def error_speech(errors: list[dict[str, Any]]) -> str:
    if not errors:
        return ""
    top = errors[0]
    joint = str(top["joint"]).replace("_", " ")
    diff = float(top["difference_deg"])
    way = "short of" if diff < 0 else "past"
    return (
        f"Your {joint} stopped approximately {abs(diff):.1f} degrees {way} "
        "its expected position. Unexpected resistance or an obstruction may be present."
    )
