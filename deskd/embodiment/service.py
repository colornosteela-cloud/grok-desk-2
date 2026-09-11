#!/usr/bin/env python3
"""Embodiment service: observe physical Teela, mirror the virtual twin."""
from __future__ import annotations

import time
from typing import Any

import robot_sim
from embodiment.encoder import encode_body, error_speech, prediction_errors
from embodiment.schema import clamp_joints, empty_health, layer
from embodiment.twin import VirtualTwin


class EmbodimentService:
    """Holds intended/expected/observed plus a virtual mirror of observed.

    Does not run servos. Does not let the twin overwrite commanded joints.
    """

    def __init__(self) -> None:
        self.twin = VirtualTwin()
        self._intended = layer("intended")
        self._expected = layer("expected")
        self._held: dict[str, str] = {}
        self._last_cmd: str = ""

    def observe(
        self,
        live_joints: dict[str, Any] | None,
        *,
        pose: str | None = None,
        motion: str | None = None,
        seq: int | None = None,
        ts: float | None = None,
        contacts: dict[str, Any] | None = None,
    ) -> None:
        """Ingest physical / MiniOS-live angles. Authoritative observed state."""
        joints = clamp_joints(live_joints)
        now = float(time.time() if ts is None else ts)
        pose_s = str(pose or self._intended.get("pose") or "home")
        motion_s = str(motion or self._intended.get("motion") or "idle")
        seq_n = int(self._intended.get("seq") or 0) if seq is None else int(seq)
        self.twin.mirror_observed(joints, pose=pose_s, motion=motion_s, seq=seq_n, ts=now)
        if isinstance(contacts, dict):
            for hand in ("left_hand", "right_hand"):
                obj = contacts.get(hand)
                if obj:
                    self._held[hand] = str(obj)
                elif hand in contacts and not obj:
                    self._held.pop(hand, None)

    def note_intent(
        self,
        cmd: dict[str, Any] | None,
        commanded_joints: dict[str, Any] | None,
        *,
        pose: str = "home",
        motion: str = "idle",
        seq: int = 0,
    ) -> None:
        """Record what Qwen/MiniOS asked for. Not a claim that it happened."""
        now = time.time()
        joints = clamp_joints(commanded_joints)
        self._intended = layer(
            "intended", joints=joints, pose=pose, motion=motion, seq=seq, ts=now
        )
        # Phase 1: expected = intended. Phase 4 replaces this with a rollout.
        self._expected = layer(
            "expected", joints=joints, pose=pose, motion=motion, seq=seq, ts=now
        )
        if isinstance(cmd, dict):
            self._last_cmd = str(cmd.get("cmd") or cmd.get("skill") or "")

    def sync_robot_state(self, state: dict[str, Any], incoming: dict[str, Any] | None = None) -> None:
        """Hook after MiniOS apply: commanded vs live, without mixing them."""
        st = state if isinstance(state, dict) else {}
        incoming = incoming if isinstance(incoming, dict) else {}
        cmd = str(incoming.get("cmd") or incoming.get("command") or "").strip().lower()
        commanded = clamp_joints(st.get("joints"))
        pose = str(st.get("pose") or "home")
        motion = str(st.get("motion") or "idle")
        seq = int(st.get("seq") or 0)
        if cmd not in {"", "status", "state", "live", "telemetry", "joints"}:
            self.note_intent(incoming, commanded, pose=pose, motion=motion, seq=seq)
        live = st.get("live") if isinstance(st.get("live"), dict) else None
        observed_joints = clamp_joints(live) if live else commanded
        self.observe(observed_joints, pose=pose, motion=motion, seq=seq)

    def snapshot(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        st = state if isinstance(state, dict) else {}
        snap = robot_sim.tracking_snapshot(st) if st else None
        commanded = clamp_joints((snap or {}).get("commanded") or st.get("joints"))
        live = clamp_joints((snap or {}).get("live") or st.get("live") or commanded)
        pose = str(st.get("pose") or self._intended.get("pose") or "home")
        motion = str(st.get("motion") or self._intended.get("motion") or "idle")
        seq = int(st.get("seq") or self._intended.get("seq") or 0)
        if st:
            self.sync_robot_state(st, {"cmd": "status"})
        observed = layer("observed", joints=live, pose=pose, motion=motion, seq=seq)
        intended = dict(self._intended)
        if st:
            intended = layer("intended", joints=commanded, pose=pose, motion=motion, seq=seq)
        expected = dict(self._expected) if self._expected.get("joints") else dict(intended)
        simulated = self.twin.snapshot()
        errors = prediction_errors(intended["joints"], observed["joints"])
        health = empty_health()
        health["emergency_stop"] = bool(st.get("estop"))
        health["motors"] = bool(st.get("motors", True))
        health["battery"] = st.get("battery")
        observed_for_encode = {
            **observed,
            "estop": health["emergency_stop"],
            "motors": health["motors"],
            "health": health,
        }
        body = encode_body(observed_for_encode, held=self._held)
        spoken = robot_sim.describe_body({**st, "joints": live, "pose": pose, "motion": motion})
        extra = error_speech(errors)
        if extra:
            spoken = (spoken.rstrip(".") + ". " + extra).strip()
        return {
            "layers": {
                "intended": intended,
                "expected": expected,
                "simulated": simulated,
                "observed": observed,
            },
            "body": body,
            "prediction_error": errors,
            "spoken": spoken,
            "rule": "REAL→VIRTUAL mirrors observed. VIRTUAL→REAL is rejected until the action gate exists.",
            "virtual_backend": self.twin.backend,
            "last_cmd": self._last_cmd,
        }

    def virtual_to_real(self, joints: dict[str, Any]) -> dict[str, Any]:
        return self.twin.virtual_drive(clamp_joints(joints))
