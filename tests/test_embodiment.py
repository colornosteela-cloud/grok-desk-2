#!/usr/bin/env python3
"""Phase 1 embodiment: physical → virtual mirror, no Qwen motor path."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deskd"))

import robot_sim  # noqa: E402
from embodiment.encoder import encode_body, prediction_errors  # noqa: E402
from embodiment.schema import BODY_ACTION_SCHEMA, JOINT_NAMES, SKILLS, body_action  # noqa: E402
from embodiment.service import EmbodimentService  # noqa: E402
from embodiment.twin import VirtualTwin  # noqa: E402


class EmbodimentPhase1Tests(unittest.TestCase):
    def test_canonical_joints_match_minios(self) -> None:
        self.assertEqual(list(JOINT_NAMES), list(robot_sim.JOINTS))
        self.assertIn("neck_pan", JOINT_NAMES)
        self.assertIn("right_elbow", JOINT_NAMES)
        self.assertIn("orient_head", SKILLS)
        self.assertIn("reach", SKILLS)
        self.assertNotIn("servo", " ".join(SKILLS))
        self.assertEqual(BODY_ACTION_SCHEMA["properties"]["skill"]["enum"], list(SKILLS))
        act = body_action("orient_head", effector="head", target="person_01")
        self.assertEqual(act["skill"], "orient_head")
        self.assertTrue(act["constraints"]["avoid_collision"])

    def test_live_observe_mirrors_virtual_not_commanded(self) -> None:
        st = robot_sim.default_state()
        robot_sim.apply(st, {"cmd": "pose", "pose": "wave"})
        commanded = dict(st["joints"])
        svc = EmbodimentService()
        svc.sync_robot_state(st, {"cmd": "pose", "pose": "wave"})
        live = dict(commanded)
        live["right_elbow"] = 90.0
        live["neck_pan"] = 12.4
        robot_sim._ingest_live(st, live)
        svc.observe(live, pose=st["pose"], motion=st["motion"], seq=st["seq"])
        snap = svc.snapshot(st)
        self.assertEqual(snap["layers"]["observed"]["joints"]["right_elbow"], 90.0)
        self.assertEqual(snap["layers"]["simulated"]["joints"]["right_elbow"], 90.0)
        self.assertEqual(snap["layers"]["intended"]["joints"]["right_elbow"], commanded["right_elbow"])
        self.assertAlmostEqual(commanded["right_elbow"], robot_sim.POSES["wave"]["right_elbow"])
        self.assertEqual(st["joints"]["right_elbow"], commanded["right_elbow"])
        errors = {row["joint"]: row for row in snap["prediction_error"]}
        self.assertIn("right_elbow", errors)
        self.assertIn("neck_pan", errors)
        self.assertIn("short of", snap["spoken"].lower())

    def test_virtual_cannot_drive_hardware(self) -> None:
        twin = VirtualTwin()
        twin.mirror_observed({"neck_pan": 10}, pose="custom")
        self.assertEqual(twin.snapshot()["joints"]["neck_pan"], 10.0)
        rejected = twin.virtual_drive({"neck_pan": 40})
        self.assertFalse(rejected["ok"])
        self.assertEqual(rejected["reason"], "virtual_to_real_requires_action_gate")
        self.assertEqual(twin.snapshot()["joints"]["neck_pan"], 10.0)

    def test_encoder_is_semantic_not_raw_dump(self) -> None:
        waved = encode_body(
            {"joints": robot_sim.POSES["wave"], "pose": "wave", "motion": "waving", "motors": True}
        )
        self.assertEqual(waved["right_arm"]["state"], "wave_hold")
        self.assertEqual(waved["balance"]["support"], "both_feet")
        self.assertTrue(waved["balance"]["stable"])
        self.assertEqual(waved["head"]["pan_deg"], 0)
        self.assertNotIn("pwm", str(waved).lower())
        self.assertNotIn("i2c", str(waved).lower())
        errs = prediction_errors({"right_elbow": 61.0}, {"right_elbow": 55.7})
        self.assertEqual(errs[0]["difference_deg"], -5.3)

    def test_identity_when_live_matches_commanded(self) -> None:
        st = robot_sim.default_state()
        robot_sim.apply(st, {"cmd": "pose", "pose": "home"})
        svc = EmbodimentService()
        svc.sync_robot_state(st, {"cmd": "pose", "pose": "home"})
        snap = svc.snapshot(st)
        self.assertEqual(snap["prediction_error"], [])
        self.assertEqual(snap["layers"]["observed"]["source"], "observed")
        self.assertEqual(snap["layers"]["simulated"]["source"], "simulated")
        self.assertEqual(snap["virtual_backend"], "kinematic")


if __name__ == "__main__":
    unittest.main()
