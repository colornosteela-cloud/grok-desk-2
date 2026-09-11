"""The HTML twin must rock the requested arm, not always the right elbow."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "ui" / "robot-simulator.html").read_text(encoding="utf-8")


class SimulatorWaveSideTests(unittest.TestCase):
    def test_wave_overlay_uses_wave_side_not_hardcoded_right(self):
        fn = re.search(r"function applyWaveFrame\([\s\S]*?\n\}", HTML)
        self.assertIsNotNone(fn, "applyWaveFrame missing")
        body = fn.group(0)
        self.assertIn("waveSide", body)
        self.assertNotIn("pose.right_elbow", body)
        self.assertNotIn("pose.right_wrist", body)

    def test_left_wave_skill_sets_wave_side_left(self):
        self.assertIn("function startWave(", HTML)
        self.assertIn("waveSide = side", HTML)
        wave = re.search(r'if\(skill === "wave"\)\{[\s\S]*?return \{status:"executing"', HTML)
        self.assertTrue(wave)
        self.assertIn("startWave(side", wave.group(0))  # type: ignore[union-attr]

    def test_greeting_gesture_honors_side(self):
        greet = re.search(r'if\(name === "greeting"\)\{[\s\S]*?return \{status:"executing"', HTML)
        self.assertIsNotNone(greet)
        self.assertIn("startWave(", greet.group(0))
        self.assertNotIn("startWaveRight(", greet.group(0))

    def test_joint_wave_hold_starts_that_side(self):
        self.assertIn("function waveHoldSide(", HTML)
        self.assertIn("startWave(holdSide", HTML)

    def test_raise_arm_drops_other_arm_and_honors_degrees(self):
        block = re.search(r'if\(skill === "raise_arm"\)\{[\s\S]*?return this\.setPose', HTML)
        self.assertIsNotNone(block)
        body = block.group(0)
        self.assertIn("parameters.degrees", body)
        self.assertIn('other+"_shoulder"] = 0', body)
        self.assertIn("stopWaving()", body)

    def test_stop_resets_lock_to_home(self):
        block = re.search(r'if\(skill === "stop"\)\{[\s\S]*?return \{status:"completed"', HTML)
        self.assertIsNotNone(block)
        self.assertIn("poses.home", block.group(0))


if __name__ == "__main__":
    unittest.main()
