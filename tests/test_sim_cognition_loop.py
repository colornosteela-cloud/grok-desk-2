"""Simulated body and cognition must share one state: act → I-feel → speech."""
from __future__ import annotations

import sys
import tempfile
import uuid
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deskd"))

import deskd as d  # noqa: E402
import robot_sim  # noqa: E402
import virtual_body  # noqa: E402


class _Bot:
    def __init__(self, tmp: str) -> None:
        self.id = "b_sim_" + uuid.uuid4().hex[:8]
        self.kind = "teela-brain"
        self.model = "qwen38-27b-q5"
        self.surface = "preview"
        self.desktop_cursor = {"x": 0, "y": 0}
        self.messages: list = []
        self.robot_state = robot_sim.default_state()
        self.workspace = Path(tmp)
        self.applied: list = []
        (self.workspace / "Desktop").mkdir(exist_ok=True)

    def apply_robot(self, body):
        self.applied.append(dict(body))
        result = robot_sim.apply(self.robot_state, body)
        return result, {"type": "desktop.action", "bot_id": self.id, "action": "robot"}

    def record_local_generation(self, *a, **k):
        return None


def _talk(bot, text, line="Okay."):
    return d.run_teela_executive_turn(
        bot, text, completer=lambda _p: {"choices": [{"message": {"content": line}}]}
    )


def _feel(bot) -> dict:
    return virtual_body.overlay(bot.id, bot.robot_state)


class SimCognitionLoopTests(unittest.TestCase):
    def test_wave_updates_simulated_body_and_speech(self):
        bot = _Bot(tempfile.mkdtemp())
        line = _talk(bot, "Wave at me.", "Waving.")
        st = _feel(bot)
        joints = st.get("live") or st.get("joints") or {}
        self.assertTrue(
            robot_sim._looks_like_wave(joints),
            f"wave hold missing from I-feel joints={joints}",
        )
        self.assertTrue(st.get("waving") or st.get("motion") == "waving" or st.get("pose") == "wave")
        low = (line or "").lower()
        self.assertIn("wav", low)
        self.assertNotIn("standing straight", low)
        self.assertNotIn("not in the wave hold", low)

    def test_hello_does_not_change_the_twin(self):
        bot = _Bot(tempfile.mkdtemp())
        _talk(bot, "Hi Teela.", "Hey!")
        st = _feel(bot)
        joints = st.get("live") or st.get("joints") or {}
        self.assertFalse(robot_sim._looks_like_wave(joints))
        self.assertNotEqual(st.get("motion"), "waving")

    def test_walk_left_replaces_stale_wave_in_ifeel(self):
        bot = _Bot(tempfile.mkdtemp())
        _talk(bot, "Wave at me.", "Waving.")
        line = _talk(bot, "Walk left", "Walking.")
        st = _feel(bot)
        self.assertIn(st.get("motion"), {"walking", "walk"})
        self.assertFalse(bool(st.get("waving")))
        low = (line or "").lower()
        self.assertTrue("walk" in low or "left" in low, line)
        self.assertNotIn("wav", low.replace("walk", ""))

    def test_look_left_after_walk_reports_the_head_not_stale_gait(self):
        bot = _Bot(tempfile.mkdtemp())
        _talk(bot, "Walk left", "Walking.")
        line = _talk(bot, "Look left", "Okay.")
        st = _feel(bot)
        joints = st.get("live") or st.get("joints") or {}
        self.assertLessEqual(float(joints.get("neck_pan") or 0), -12)
        low = (line or "").lower()
        self.assertIn("left", low)
        self.assertIn("look", low)
        self.assertNotIn("walking", low)

    def test_look_left_turns_head_and_says_so(self):
        bot = _Bot(tempfile.mkdtemp())
        line = _talk(bot, "Look left", "Okay.")
        st = _feel(bot)
        joints = st.get("live") or st.get("joints") or {}
        self.assertLessEqual(float(joints.get("neck_pan") or 0), -12)
        self.assertNotEqual(st.get("motion"), "walking")
        low = (line or "").lower()
        self.assertIn("left", low)
        self.assertNotIn("standing straight", low)
        self.assertNotIn("walking", low)

    def test_walk_left_speech_uses_user_direction(self):
        bot = _Bot(tempfile.mkdtemp())
        line = _talk(bot, "Walk left", "Okay.")
        st = _feel(bot)
        self.assertIn(st.get("motion"), {"walking", "walk"})
        self.assertIn("left", (line or "").lower())
        self.assertNotIn("east", (line or "").lower())

    def test_left_hand_wave_is_left_hold_and_confirmed(self):
        bot = _Bot(tempfile.mkdtemp())
        line = _talk(bot, "Wave with your left hand", "Okay.")
        st = _feel(bot)
        joints = st.get("live") or st.get("joints") or {}
        self.assertTrue(robot_sim._looks_like_wave(joints, "left"), joints)
        self.assertFalse(robot_sim._looks_like_wave(joints, "right"))
        low = (line or "").lower()
        self.assertIn("wav", low)
        self.assertIn("left", low)
        self.assertNotIn("not in the wave hold", low)

    def test_known_wave_does_not_call_the_llm(self):
        bot = _Bot(tempfile.mkdtemp())

        def boom(_payload):
            raise AssertionError("known wave must not call the LLM")

        line = d.run_teela_executive_turn(bot, "Wave with your left hand", completer=boom)
        self.assertIn("wav", (line or "").lower())
        self.assertNotIn("tool_call", (line or "").lower())
        joints = (_feel(bot).get("live") or _feel(bot).get("joints") or {})
        self.assertTrue(robot_sim._looks_like_wave(joints, "left"), joints)

    def test_leaked_robot_joint_xml_is_not_spoken(self):
        bot = _Bot(tempfile.mkdtemp())
        xml = (
            "<tool_call><function=bot_desktop__robot_joint>"
            '<parameter=joints>{"left_shoulder": 24, "left_elbow": 118, "neck_pan": -25}'
            "</parameter></function></tool_call>"
        )
        line = _talk(bot, "Wave with your left hand", xml)
        self.assertNotIn("tool_call", (line or "").lower())
        self.assertNotIn("robot_joint", (line or "").lower())
        self.assertIn("left", (line or "").lower())
        joints = (_feel(bot).get("live") or {})
        self.assertEqual(round(float(joints.get("neck_pan") or 0), 1), 0.0)


if __name__ == "__main__":
    unittest.main()
