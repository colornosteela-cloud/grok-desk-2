#!/usr/bin/env python3
"""Kinematic Virtual Teela. REAL→VIRTUAL only. MuJoCo plant is a later backend."""
from __future__ import annotations

import time
from typing import Any

from embodiment.schema import clamp_joints, layer


class VirtualTwin:
    """Always-on digital twin. Phase 1: kinematic mirror of observed joints.

    MuJoCo (same joint names, limits, zeros) plugs in behind `backend`
    without changing observe() or snapshot().
    """

    def __init__(self) -> None:
        self.backend = "kinematic"
        self._joints = clamp_joints(None)
        self._pose = "home"
        self._motion = "idle"
        self._seq = 0
        self._ts = time.time()

    def mirror_observed(
        self,
        joints: dict[str, float],
        *,
        pose: str = "home",
        motion: str = "idle",
        seq: int = 0,
        ts: float | None = None,
    ) -> None:
        """REAL → VIRTUAL. The only write from physical truth."""
        self._joints = clamp_joints(joints)
        self._pose = str(pose or "home")
        self._motion = str(motion or "idle")
        self._seq = int(seq or 0)
        self._ts = float(time.time() if ts is None else ts)

    def snapshot(self) -> dict[str, Any]:
        return layer(
            "simulated",
            joints=self._joints,
            pose=self._pose,
            motion=self._motion,
            seq=self._seq,
            ts=self._ts,
        )

    def virtual_drive(self, _joints: dict[str, float]) -> dict[str, Any]:
        """VIRTUAL → REAL is forbidden here. Action gate comes in Phase 3."""
        return {
            "ok": False,
            "status": "rejected",
            "reason": "virtual_to_real_requires_action_gate",
            "backend": self.backend,
        }
