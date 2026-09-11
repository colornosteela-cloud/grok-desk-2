"""Authoritative BodyState: browser is a view, not the owner."""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deskd"))

import body_state  # noqa: E402
import virtual_body  # noqa: E402


class BodyStateAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.store = body_state.BodyStateStore(self.tmp / "body.sqlite", bot_id="b_body")

    def test_commanded_shortfall_is_actual_not_success(self) -> None:
        self.store.apply_commanded({"right_elbow": 90.0, "right_shoulder": 142.0}, pose="custom")
        self.store.apply_measured({"right_elbow": 78.1, "right_shoulder": 142.0}, pose="custom", source="simulation")
        snap = self.store.snapshot()
        el = snap["joints"]["right_elbow"]
        self.assertAlmostEqual(float(el["commanded"]), 90.0, places=1)
        self.assertAlmostEqual(float(el["actual"]), 78.1, places=1)
        self.assertGreater(abs(float(el["error"])), 5)
        self.assertFalse(snap.get("target_reached"))
        compact = body_state.compact_block(snap)
        self.assertIn("78", compact)
        self.assertIn("90", compact)
        self.assertNotIn("I think", compact)

    def test_stale_revision_is_rejected(self) -> None:
        first = self.store.apply_measured({"right_shoulder": 40.0}, pose="custom")
        self.store.apply_measured({"right_shoulder": 50.0}, pose="custom")
        ok = self.store.apply_measured(
            {"right_shoulder": 10.0}, pose="custom", revision=int(first["revision"])
        )
        self.assertFalse(ok.get("accepted"))
        self.assertAlmostEqual(float(self.store.snapshot()["joints"]["right_shoulder"]["actual"]), 50.0, places=1)

    def test_snapshot_survives_reload(self) -> None:
        self.store.apply_commanded({"right_shoulder": 142.0, "right_elbow": 18.0}, pose="custom", last_action="raise_arm")
        self.store.apply_measured({"right_shoulder": 142.0, "right_elbow": 18.0}, pose="custom")
        rev = int(self.store.snapshot()["revision"])
        again = body_state.BodyStateStore(self.tmp / "body.sqlite", bot_id="b_body")
        snap = again.snapshot()
        self.assertEqual(int(snap["revision"]), rev)
        self.assertAlmostEqual(float(snap["joints"]["right_shoulder"]["actual"]), 142.0, places=1)
        self.assertEqual(snap.get("last_action"), "raise_arm")

    def test_browser_reconnect_does_not_home_a_raised_arm(self) -> None:
        bid = "b_body"
        self.store.apply_commanded({"right_shoulder": 142.0, "right_elbow": 18.0, "left_shoulder": 0.0}, pose="custom")
        self.store.apply_measured({"right_shoulder": 142.0, "right_elbow": 18.0, "left_shoulder": 0.0}, pose="custom")
        restored = body_state.restore_overlay(bid, self.store.snapshot())
        virtual_body.note_state(
            bid,
            {
                "mode": "virtual",
                "pose": "home",
                "motion": "idle",
                "waving": False,
                "joints": {"right_shoulder": 0, "right_elbow": 0, "left_shoulder": 0, "left_elbow": 0},
            },
        )
        live = virtual_body.latest_state(bid)
        sh = float((live.get("joints") or live.get("live") or {}).get("right_shoulder") or 0)
        self.assertGreaterEqual(sh, 100, (restored, live))
        self.assertNotEqual(str(live.get("pose") or ""), "home")

    def test_history_records_meaningful_snapshots_not_every_tick(self) -> None:
        self.store.apply_measured({"right_shoulder": 10.0}, pose="custom")
        for _ in range(20):
            self.store.apply_measured({"right_shoulder": 10.0}, pose="custom")
        hist = self.store.history(limit=50)
        self.assertLessEqual(len(hist), 3)

    def test_ui_hydrate_reads_authoritative_body_state(self) -> None:
        app = (ROOT / "ui" / "app.js").read_text(encoding="utf-8")
        self.assertIn("/body/state", app)
        self.assertIn("hydrateRobotIframe", app)


if __name__ == "__main__":
    unittest.main()
