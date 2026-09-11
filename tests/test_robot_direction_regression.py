"""Direction regressions at the simulator/virtual-body seam; no hardware."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deskd"))
import robot_sim
import virtual_body


class RobotDirectionRegressionTests(unittest.TestCase):
    def test_walk_back_means_backward_not_in_place(self):
        for text in ("walk back", "please walk back", "walk backward", "walk backwards"):
            with self.subTest(text=text):
                cmd = robot_sim.infer_command(text)
                self.assertEqual(cmd, {"cmd": "walk", "direction": "back"})
                self.assertEqual(virtual_body.teela_args_from_motor(cmd, text),
                                 ("bot_desktop__robot_motion", cmd))

    def test_lower_named_arm_keeps_anatomical_side(self):
        for side in ("left", "right"):
            for noun in ("arm", "hand"):
                with self.subTest(side=side, noun=noun):
                    text = f"lower your {side} {noun}"
                    hit = virtual_body.teela_args_from_motor(robot_sim.infer_command(text), text)
                    self.assertEqual(hit, ("bot_desktop__teela_body_action",
                                           {"skill": "lower_arm", "side": side}))

    def test_deduplication_respects_sided_gestures_and_default_right(self):
        for skill in ("wave", "raise_arm", "lower_arm", "greeting", "point"):
            with self.subTest(skill=skill):
                want = {"skill": skill, "side": "left"}
                self.assertFalse(virtual_body.same_request(
                    {"skill": skill, "side": "right"}, want))
                self.assertFalse(virtual_body.same_request({"skill": skill}, want))
                self.assertTrue(virtual_body.same_request(
                    {"skill": skill, "parameters": {"side": "left"}}, want))

    def test_wave_on_other_side_does_not_satisfy_request(self):
        for wanted, moving in (("left", "right"), ("right", "left")):
            for skill in ("wave", "greeting"):
                with self.subTest(wanted=wanted, skill=skill):
                    state = {"motion": "waving", "joints": {
                        f"{moving}_shoulder": 24, f"{moving}_elbow": 118}}
                    self.assertTrue(virtual_body.move_still_needed(
                        {"skill": skill, "side": wanted}, state))
                    self.assertFalse(virtual_body.move_still_needed(
                        {"skill": skill, "side": moving}, state))


if __name__ == "__main__":
    unittest.main()
