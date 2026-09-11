#!/usr/bin/env python3
"""Virtual Teela body: Qwen tools → HTML simulator, no hardware."""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deskd"))

import action_orchestrator as orch  # noqa: E402
import deskd as d  # noqa: E402
import robot_sim  # noqa: E402
import virtual_body  # noqa: E402


class VirtualBodyToolTests(unittest.TestCase):
    def test_turn_and_walk_left_is_walk_not_look(self) -> None:
        intent = "Can you turn and walk left?"
        cmd = robot_sim.infer_command(intent)
        self.assertEqual(cmd.get("cmd"), "walk")
        self.assertEqual(cmd.get("direction"), "left")
        hit = virtual_body.teela_args_from_motor(cmd, intent)
        self.assertEqual(hit[0], "bot_desktop__robot_motion")
        self.assertEqual(hit[1]["cmd"], "walk")
        self.assertEqual(hit[1]["direction"], "left")
        raw = d.local_llm_direct_completion({"messages": [{"role": "user", "content": intent}]})
        call = json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "bot_desktop__robot_motion")
        self.assertEqual(json.loads(call["arguments"])["cmd"], "walk")
        self.assertEqual(d.speech_after_motor({"messages": [{"role": "user", "content": intent}]}), "I'm walking left.")

    def test_correction_walk_left_not_blocked_by_dont(self) -> None:
        intent = "No, I don't want you to look left, I want you to walk left"
        self.assertTrue(robot_sim.looks_like_motor(intent))
        self.assertFalse(robot_sim._blocked_by_negation(intent))
        cmd = robot_sim.infer_command(intent)
        self.assertEqual(cmd.get("cmd"), "walk")
        self.assertEqual(cmd.get("direction"), "left")
        hit = virtual_body.teela_args_from_motor(cmd, intent)
        self.assertEqual(hit[0], "bot_desktop__robot_motion")
        self.assertEqual(hit[1]["direction"], "left")
        hint = d.teela_capability_hint(intent)
        self.assertTrue(not hint or "body" in hint.lower())

    def test_head_left_maps_to_html_negative_pan(self) -> None:
        intent = "Teela, turn your head to the left."
        cmd = robot_sim.infer_command(intent)
        self.assertIsNotNone(cmd)
        hit = virtual_body.teela_args_from_motor(cmd, intent)
        self.assertIsNotNone(hit)
        name, args = hit
        self.assertEqual(name, "bot_desktop__teela_body_action")
        self.assertEqual(args["skill"], "orient_head")
        self.assertEqual(args["pan_deg"], -25.0)

    def test_wave_right_reuses_wave_skill(self) -> None:
        intent = "Teela, wave your right hand."
        cmd = robot_sim.infer_command(intent)
        self.assertIsNotNone(cmd)
        hit = virtual_body.teela_args_from_motor(cmd, intent)
        self.assertIsNotNone(hit)
        name, args = hit
        self.assertEqual(name, "bot_desktop__teela_body_action")
        self.assertEqual(args["skill"], "wave")
        self.assertEqual(args["side"], "right")

    def test_direct_completion_emits_teela_orient_head(self) -> None:
        raw = d.local_llm_direct_completion(
            {"messages": [{"role": "user", "content": "Teela, turn your head to the left."}]}
        )
        self.assertIsNotNone(raw)
        call = json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "bot_desktop__teela_body_action")
        args = json.loads(call["arguments"])
        self.assertEqual(args["skill"], "orient_head")
        self.assertEqual(args["pan_deg"], -25.0)

    def test_direct_completion_emits_teela_wave(self) -> None:
        raw = d.local_llm_direct_completion(
            {"messages": [{"role": "user", "content": "Teela, wave your right hand."}]}
        )
        self.assertIsNotNone(raw)
        call = json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "bot_desktop__teela_body_action")
        args = json.loads(call["arguments"])
        self.assertEqual(args["skill"], "wave")
        self.assertEqual(args["side"], "right")

    def test_fallback_motor_still_uses_robot_pose(self) -> None:
        raw = d.fallback_motor_completion(
            {"messages": [{"role": "user", "content": "can you wave"}]}
        )
        self.assertIsNotNone(raw)
        call = json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "bot_desktop__robot_pose")
        args = json.loads(call["arguments"])
        self.assertEqual(args["pose"], "wave")

    def test_canonicalize_teela_short_name(self) -> None:
        self.assertEqual(
            d.canonicalize_tool_name("teela_body_action", None),
            "bot_desktop__teela_body_action",
        )
        self.assertEqual(
            d.canonicalize_tool_name("mcp__bot_desktop__teela_stop", None),
            "bot_desktop__teela_stop",
        )

    def test_unknown_skill_rejected(self) -> None:
        out = virtual_body.submit_action("b_vb_unknown", "dance", {})
        self.assertEqual(out["status"], "rejected")
        self.assertIn("unknown_skill", str(out.get("reason") or ""))

    def test_action_returns_executing_immediately(self) -> None:
        t0 = __import__("time").monotonic()
        out = virtual_body.submit_action(
            "b_vb_fast", "orient_head", {"pan_deg": -25}
        )
        elapsed = __import__("time").monotonic() - t0
        self.assertEqual(out["status"], "executing")
        self.assertTrue(out.get("started"))
        self.assertLess(elapsed, 0.25)

    def test_qwen_can_move_while_pose_locked(self) -> None:
        virtual_body.note_state(
            "b_vb_lock", {"mode": "virtual", "locked": True, "joints": {"neck_pan": 0}}
        )
        out = virtual_body.submit_action(
            "b_vb_lock", "orient_head", {"pan_deg": -25}
        )
        self.assertEqual(out["status"], "executing")
        self.assertNotEqual(out.get("reason"), "body_locked")

    def test_raise_right_arm_keeps_teela_right(self) -> None:
        intent = "Raise your right arm."
        hit = virtual_body.teela_args_from_motor(robot_sim.infer_command(intent), intent)
        self.assertEqual(hit[0], "bot_desktop__teela_body_action")
        self.assertEqual(hit[1]["skill"], "raise_arm")
        self.assertEqual(hit[1]["side"], "right")

    def test_look_straight_centers_head(self) -> None:
        intent = "look straight"
        hit = virtual_body.teela_args_from_motor(robot_sim.infer_command(intent), intent)
        self.assertEqual(hit[0], "bot_desktop__teela_body_action")
        self.assertEqual(hit[1]["skill"], "orient_head")
        self.assertEqual(hit[1]["pan_deg"], 0.0)
        self.assertEqual(hit[1]["tilt_deg"], 0.0)
        cmd = robot_sim.infer_command(intent)
        self.assertEqual(cmd["joint"], "neck_pan")
        self.assertEqual(cmd["value"], 0)

    def test_head_speech_matches_live_pan_sign(self) -> None:
        self.assertEqual(virtual_body.head_speech({"neck_pan": -25}), "I'm looking left.")
        self.assertEqual(virtual_body.head_speech({"neck_pan": 25}), "I'm looking right.")
        self.assertEqual(virtual_body.head_speech({"neck_pan": 0}), "I'm looking straight ahead.")

    def test_empty_joints_are_not_already_looking_straight(self) -> None:
        self.assertTrue(
            virtual_body.move_still_needed(
                {"skill": "orient_head", "pan_deg": 0.0, "tilt_deg": 0.0},
                {"joints": {}},
            )
        )
        self.assertTrue(
            virtual_body.move_still_needed(
                {"skill": "orient_head", "pan_deg": 0.0},
                {},
            )
        )
        self.assertFalse(
            virtual_body.move_still_needed(
                {"skill": "orient_head", "pan_deg": 0.0, "tilt_deg": 0.0},
                {"joints": {"neck_pan": 0, "neck_tilt": 0}},
            )
        )
        self.assertTrue(
            virtual_body.move_still_needed(
                {"skill": "orient_head", "pan_deg": 0.0},
                {"joints": {"neck_pan": -25}},
            )
        )

    def test_wait_settle_idle_is_immediate(self) -> None:
        t0 = __import__("time").monotonic()
        virtual_body.wait_settle("b_vb_idle_settle", timeout=0.8)
        self.assertLess(__import__("time").monotonic() - t0, 0.2)

    def test_speech_uses_live_pan_not_the_request(self) -> None:
        virtual_body.note_state(
            "b_vb_live_pan",
            {"mode": "virtual", "joints": {"neck_pan": -25, "neck_tilt": 0}},
        )

        class _Bot:
            id = "b_vb_live_pan"
            robot_state = {"joints": {"neck_pan": 0}}

        text = d.speech_after_motor(
            {"messages": [{"role": "user", "content": "look straight"}]},
            bot=_Bot(),
        )
        self.assertEqual(text, "I'm looking left.")

    def test_look_straight_emits_pan_zero_while_facing_left(self) -> None:
        virtual_body.note_state(
            "b_vb_need_center",
            {"mode": "virtual", "joints": {"neck_pan": -25, "neck_tilt": 0}},
        )

        class _Bot:
            id = "b_vb_need_center"
            robot_state = {"pose": "wave", "joints": {"neck_pan": 0}}

        raw = d.local_llm_direct_completion(
            {"messages": [{"role": "user", "content": "look straight"}]},
            bot=_Bot(),
        )
        self.assertIsNotNone(raw)
        args = json.loads(json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args["skill"], "orient_head")
        self.assertEqual(args["pan_deg"], 0.0)
        self.assertEqual(args["tilt_deg"], 0.0)

    def test_wrong_head_tool_does_not_skip_look_straight(self) -> None:
        virtual_body.note_state(
            "b_vb_wrong_head",
            {"mode": "virtual", "joints": {"neck_pan": -25, "neck_tilt": 0}},
        )

        class _Bot:
            id = "b_vb_wrong_head"
            robot_state = {"joints": {"neck_pan": -25}}

        payload = {
            "messages": [
                {"role": "user", "content": "look straight"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "bot_desktop__teela_body_action",
                                "arguments": '{"skill":"orient_head","pan_deg":-25}',
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "bot_desktop__teela_body_action",
                    "content": json.dumps(
                        {"status": "executing", "skill": "orient_head", "started": True}
                    ),
                },
            ]
        }
        raw = d.local_llm_direct_completion(payload, bot=_Bot())
        self.assertIsNotNone(raw)
        msg = json.loads(raw)["choices"][0]["message"]
        self.assertTrue(msg.get("tool_calls"))
        args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args["skill"], "orient_head")
        self.assertEqual(args["pan_deg"], 0.0)

    def test_settle_timeout_waits_for_look_not_wave(self) -> None:
        self.assertGreater(virtual_body.settle_timeout("orient_head"), 1.0)
        self.assertGreater(virtual_body.settle_timeout("raise_arm"), 1.0)
        self.assertEqual(virtual_body.settle_timeout("wave"), 0.0)
        self.assertEqual(virtual_body.settle_timeout("gesture"), 0.0)

    def test_look_up_a_little(self) -> None:
        intent = "Look up a little."
        hit = virtual_body.teela_args_from_motor(robot_sim.infer_command(intent), intent)
        self.assertEqual(hit[1]["skill"], "orient_head")
        self.assertGreater(hit[1]["tilt_deg"], 0)
        self.assertLessEqual(hit[1]["tilt_deg"], 12)

    def test_lower_arm(self) -> None:
        intent = "Put your arm back down."
        hit = virtual_body.teela_args_from_motor(robot_sim.infer_command(intent), intent)
        self.assertEqual(hit[1]["skill"], "lower_arm")

    def test_toward_right_shoulder(self) -> None:
        intent = "Turn your head toward your right shoulder."
        hit = virtual_body.teela_args_from_motor(robot_sim.infer_command(intent), intent)
        self.assertEqual(hit[1]["skill"], "orient_head")
        self.assertEqual(hit[1]["pan_deg"], 25.0)

    def test_wave_hello_is_greeting_gesture(self) -> None:
        intent = "Teela, wave hello."
        hit = virtual_body.teela_args_from_motor(robot_sim.infer_command(intent), intent)
        self.assertEqual(hit[0], "bot_desktop__teela_gesture")
        self.assertEqual(hit[1]["gesture"], "greeting")

    def test_direct_wave_hello_emits_gesture(self) -> None:
        raw = d.local_llm_direct_completion(
            {"messages": [{"role": "user", "content": "Teela, wave hello."}]}
        )
        call = json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "bot_desktop__teela_gesture")
        self.assertEqual(json.loads(call["arguments"])["gesture"], "greeting")

    def test_executing_result_counts_as_moved(self) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": "Teela, wave hello."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "bot_desktop__teela_gesture",
                                "arguments": '{"gesture":"greeting"}',
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "bot_desktop__teela_gesture",
                    "content": json.dumps({"status": "executing", "started": True, "skill": "gesture"}),
                },
            ]
        }
        self.assertTrue(d.payload_already_moved(payload))
        speech = json.loads(d.local_llm_direct_completion(payload))
        self.assertIn("hi", speech["choices"][0]["message"]["content"].lower())

    def test_look_left_not_skipped_after_wave_pose(self) -> None:
        class _Bot:
            id = "b_vb_wave"
            robot_state = {
                "pose": "wave",
                "motion": "waving",
                "joints": dict(robot_sim.POSES["wave"]),
            }

        payload = {"messages": [{"role": "user", "content": "Teela, look to your left."}]}
        raw = d.local_llm_direct_completion(payload, bot=_Bot())
        self.assertIsNotNone(raw)
        call = json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "bot_desktop__teela_body_action")
        self.assertEqual(json.loads(call["arguments"])["skill"], "orient_head")
        self.assertEqual(json.loads(call["arguments"])["pan_deg"], -25.0)

    def test_mixed_while_is_parallel_not_body_only(self) -> None:
        intent = "While you're checking it, look left."
        hit = orch.classify(intent)
        self.assertEqual(hit["mode"], "parallel")
        self.assertTrue(hit["body"])
        self.assertTrue(hit["agent"])
        payload = {"messages": [{"role": "user", "content": intent}]}
        self.assertFalse(d.local_llm_motor_turn(payload))
        self.assertIsNone(d.local_llm_direct_completion(payload))
        out = d.rewrite_local_llm_chat_payload(
            {
                "model": "qwen38",
                "messages": [{"role": "user", "content": intent}],
                "tools": [
                    {"type": "function", "function": {"name": "shell"}},
                    {"type": "function", "function": {"name": "bot_desktop__teela_body_action"}},
                ],
            }
        )
        names = [t["function"]["name"] for t in out["tools"]]
        self.assertIn("bot_desktop__teela_body_action", names)
        self.assertIn("shell", names)

    def test_wave_when_finished_is_after(self) -> None:
        intent = "inspect your system information and wave when you're finished"
        self.assertEqual(orch.classify(intent)["mode"], "after")
        self.assertIsNone(
            d.local_llm_direct_completion({"messages": [{"role": "user", "content": intent}]})
        )

    def test_inspect_then_wave_runs_without_llama(self) -> None:
        intent = "inspect your system information and wave when you're finished"
        text = orch.inspect_then_wave("b_vb_insp", intent)
        self.assertIsNotNone(text)
        self.assertIn("waving now", (text or "").lower())
        raw = d.local_llm_direct_completion(
            {"messages": [{"role": "user", "content": intent}]},
            bot=type("B", (), {"id": "b_vb_insp2", "robot_state": {}})(),
        )
        self.assertIsNotNone(raw)
        msg = json.loads(raw)["choices"][0]["message"]
        self.assertIn("waving now", (msg.get("content") or "").lower())

    def test_inspect_and_wave_not_swallowed_by_leftover_wave_pose(self) -> None:
        intent = "Teela, inspect your system information and wave when you're finished."
        payload = {"messages": [{"role": "user", "content": intent}]}

        class _Bot:
            id = "b_vb_after"
            robot_state = {
                "pose": "wave",
                "motion": "waving",
                "joints": dict(robot_sim.POSES["wave"]),
            }

        self.assertFalse(d.motor_already_satisfied(payload, _Bot()))
        self.assertFalse(d.payload_already_moved(payload, _Bot()))
        raw = d.local_llm_direct_completion(payload, bot=_Bot())
        self.assertIsNotNone(raw)
        self.assertIn("waving now", json.loads(raw)["choices"][0]["message"]["content"].lower())
        out = d.rewrite_local_llm_chat_payload(
            {
                "model": "qwen38",
                "messages": [{"role": "user", "content": intent}],
                "tools": [{"type": "function", "function": {"name": "shell"}}],
            },
            bot=_Bot(),
        )
        names = [t["function"]["name"] for t in out.get("tools") or []]
        self.assertIn("shell", names)
        self.assertIn("bot_desktop__teela_body_action", names)

    def test_look_left_stays_body_only(self) -> None:
        self.assertEqual(orch.classify("Can you look left")["mode"], "body")
        self.assertTrue(
            d.local_llm_motor_turn({"messages": [{"role": "user", "content": "Can you look left"}]})
        )

    def test_activity_snapshot_lists_body_task(self) -> None:
        orch.note("b_act", "body", "orient_head", status="executing", id="act_test")
        snap = orch.snapshot("b_act")
        self.assertTrue(any("moving" in x for x in snap["you_are"]))

    def test_stop_and_walk_still_plan(self) -> None:
        raw = d.local_llm_direct_completion(
            {"messages": [{"role": "user", "content": "can you stop and walk left"}]}
        )
        call = json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "bot_desktop__robot_motion")

    def test_teela_result_counts_as_already_moved(self) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": "Teela, turn your head to the left."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "bot_desktop__teela_body_action",
                                "arguments": '{"skill":"orient_head","pan_deg":-25}',
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "bot_desktop__teela_body_action",
                    "content": json.dumps(
                        {
                            "status": "completed",
                            "skill": "orient_head",
                            "body_state": {"head_pan": -25, "joints": {"neck_pan": -25}},
                        }
                    ),
                },
            ]
        }
        self.assertTrue(d.payload_already_moved(payload))
        self.assertTrue(
            d.robot_args_usable(
                "bot_desktop__teela_body_action",
                {"skill": "orient_head", "pan_deg": -25},
            )
        )

    def test_live_telemetry_clears_stale_wave_label(self) -> None:
        state = {
            "pose": "wave",
            "motion": "waving",
            "waving": True,
            "joints": dict(robot_sim.POSES["wave"]),
            "live": dict(robot_sim.POSES["wave"]),
        }
        self.assertFalse(robot_sim.observed_waving(state))
        self.assertTrue(robot_sim._looks_like_wave(state["live"]))
        robot_sim.apply(
            state,
            {
                "cmd": "live",
                "pose": "home",
                "motion": "idle",
                "waving": False,
                "live": dict(robot_sim.POSES["home"]),
            },
        )
        self.assertEqual(state.get("pose"), "home")
        self.assertEqual(state.get("motion"), "idle")
        self.assertFalse(state.get("waving"))
        self.assertFalse(robot_sim.observed_waving(state))
        spoken = robot_sim.describe_body(state, which="live").lower()
        self.assertNotIn("i'm waving", spoken)
        sense = d.proprioception_block(state)
        self.assertIn("Wave: not waving.", sense)
        self.assertNotIn("Wave: yes —", sense)

    def test_html_standing_overlay_beats_minios_wave(self) -> None:
        virtual_body.note_state(
            "b_vb_false_wave",
            {
                "mode": "virtual",
                "pose": "home",
                "motion": "idle",
                "waving": False,
                "joints": dict(robot_sim.POSES["home"]),
            },
        )
        minios = {
            "pose": "wave",
            "motion": "waving",
            "joints": dict(robot_sim.POSES["wave"]),
        }
        st = virtual_body.overlay("b_vb_false_wave", minios)
        self.assertEqual(st.get("pose"), "home")
        self.assertEqual(st.get("motion"), "idle")
        self.assertFalse(st.get("waving"))
        self.assertFalse(robot_sim.observed_waving(st))

        class _Bot:
            id = "b_vb_false_wave"
            robot_state = minios

        text = d.speech_after_motor(
            {
                "messages": [
                    {"role": "user", "content": "wave at me"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "bot_desktop__robot_pose",
                                    "arguments": '{"pose":"wave"}',
                                }
                            }
                        ],
                    },
                ]
            },
            bot=_Bot(),
        )
        self.assertNotEqual(text, "I'm waving.")
        self.assertIn("not waving", text.lower())

    def test_stale_wave_pose_does_not_skip_a_new_wave(self) -> None:
        class _Bot:
            id = "b_vb_stale_wave"
            robot_state = {
                "pose": "wave",
                "motion": "waving",
                "joints": dict(robot_sim.POSES["home"]),
                "live": dict(robot_sim.POSES["home"]),
                "waving": False,
            }

        payload = {"messages": [{"role": "user", "content": "Can you wave?"}]}
        self.assertFalse(d.motor_already_satisfied(payload, _Bot()))
        self.assertFalse(d.payload_already_moved(payload, _Bot()))
        raw = d.local_llm_direct_completion(payload, bot=_Bot())
        self.assertIsNotNone(raw)
        call = json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertIn("teela_body_action", call["name"])
        self.assertEqual(json.loads(call["arguments"]).get("skill"), "wave")

    def test_minios_chat_wave_skips_llama(self) -> None:
        class _Bot:
            id = "b_vb_chat_wave"
            kind = "teela-brain"
            model = "qwen38-27b-q5"
            robot_state = robot_sim.default_state()
            messages: list = []
            dispatched: list = []

            def apply_robot(self, body):
                return {"ok": True, "pose": body.get("pose")}, {}

        bot = _Bot()
        rounds = [
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "bot_desktop__teela_body_action",
                                        "arguments": '{"skill":"wave","side":"right"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "Waving."}}]},
        ]

        def fake_complete(_payload):
            return rounds.pop(0)

        def capture(_bot, name, args=None, **_kwargs):
            bot.dispatched.append((name, args))
            return {"ok": True, "status": "executing", "skill": "wave"}

        with patch.object(d, "execute_teela_allowed_tool", side_effect=capture):
            line = d.run_teela_minios_turn(bot, "Can you wave?", completer=fake_complete)
        self.assertTrue(bot.dispatched)
        self.assertIn("wave", str(bot.dispatched).lower())
        self.assertIn("wav", (line or "").lower())

    def test_hi_can_you_wave_dispatches_despite_leftover_wave(self) -> None:
        class _Bot:
            id = "b_vb_hi_wave"
            kind = "teela-brain"
            model = "qwen38-27b-q5"
            robot_state = {
                "pose": "wave",
                "motion": "waving",
                "waving": True,
                "joints": dict(robot_sim.POSES["wave"]),
                "live": dict(robot_sim.POSES["wave"]),
            }
            dispatched: list = []

            def apply_robot(self, body):
                return {"ok": True}, {}

        bot = _Bot()

        def capture(_bot, name, args=None, **_kwargs):
            bot.dispatched.append((name, args))
            return {"ok": True, "status": "executing", "skill": "wave"}

        rounds = [
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "bot_desktop__teela_body_action",
                                        "arguments": '{"skill":"wave","side":"right"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "Waving."}}]},
        ]

        def fake_complete(_payload):
            return rounds.pop(0)

        with patch.object(d, "execute_teela_allowed_tool", side_effect=capture):
            line = d.run_teela_minios_turn(bot, "Hi, can you wave?", completer=fake_complete)
        self.assertTrue(bot.dispatched)
        self.assertIn("wave", str(bot.dispatched).lower() + str(line).lower())
        line2 = None
        bot.dispatched.clear()
        rounds2 = [
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c2",
                                    "function": {
                                        "name": "bot_desktop__teela_body_action",
                                        "arguments": '{"skill":"wave","side":"right"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "Waving at you."}}]},
        ]

        def fake_complete2(_payload):
            return rounds2.pop(0)

        with patch.object(d, "execute_teela_allowed_tool", side_effect=capture):
            line2 = d.run_teela_minios_turn(bot, "Can you wave at me?", completer=fake_complete2)
        self.assertTrue(bot.dispatched)
        self.assertTrue(line2)


class VirtualStampPersistenceTests(unittest.TestCase):
    def tearDown(self) -> None:
        virtual_body.note_state(
            "b_stamp",
            {"mode": "virtual", "joints": {}, "motion": "idle", "waving": False, "stamp_t": time.time()},
        )

    def test_html_echo_cannot_clobber_fresh_stamp(self) -> None:
        st = virtual_body.stamp_intended("b_stamp", "raise_arm", {"side": "right"})
        self.assertGreaterEqual(float((st.get("joints") or {}).get("right_shoulder") or 0), 100)
        virtual_body.note_state(
            "b_stamp",
            {
                "mode": "virtual",
                "joints": {"right_shoulder": 0, "right_elbow": 5, "left_shoulder": 0, "left_elbow": 5},
                "motion": "idle",
                "waving": False,
            },
        )
        live = virtual_body.latest_state("b_stamp")
        sh = float((live.get("joints") or live.get("live") or {}).get("right_shoulder") or 0)
        self.assertGreaterEqual(sh, 100, live)

    def test_html_echo_cannot_clobber_stamp_after_delay(self) -> None:
        virtual_body.stamp_intended("b_stamp", "stop", {})
        time.sleep(0.05)
        virtual_body.note_state(
            "b_stamp",
            {
                "mode": "virtual",
                "pose": "custom",
                "joints": {"right_shoulder": 142, "right_elbow": 18, "left_shoulder": 0, "left_elbow": 5},
                "motion": "idle",
                "waving": False,
            },
        )
        live = virtual_body.latest_state("b_stamp")
        sh = float((live.get("joints") or live.get("live") or {}).get("right_shoulder") or 0)
        self.assertLess(sh, 25, live)

    def test_stop_homes_a_raised_arm(self) -> None:
        virtual_body.stamp_intended("b_stamp", "raise_arm", {"side": "right"})
        st = virtual_body.stamp_intended("b_stamp", "stop", {})
        joints = st.get("joints") or {}
        self.assertLess(float(joints.get("right_shoulder") or 0), 25, joints)
        self.assertFalse(st.get("waving"))

    def test_html_action_result_cannot_clobber_fresh_stamp(self) -> None:
        result = virtual_body.submit_action("b_stamp", "raise_arm", {"side": "right"}, timeout=0)
        aid = str(result.get("action_id") or "")
        self.assertTrue(aid)
        virtual_body.handle_client_message(
            "b_stamp",
            {
                "type": "action_result",
                "action_id": aid,
                "status": "completed",
                "skill": "raise_arm",
                "body_state": {
                    "mode": "virtual",
                    "pose": "wave",
                    "motion": "waving",
                    "waving": True,
                    "joints": {
                        "left_shoulder": 24,
                        "left_elbow": 118,
                        "right_shoulder": 142,
                        "right_elbow": 18,
                    },
                },
            },
        )
        live = virtual_body.latest_state("b_stamp")
        joints = live.get("joints") or live.get("live") or {}
        self.assertFalse(live.get("waving"), live)
        self.assertLess(float(joints.get("left_shoulder") or 0), 15, live)
        self.assertGreaterEqual(float(joints.get("right_shoulder") or 0), 100, live)

    def test_stale_minios_live_cannot_undo_stop_stamp(self) -> None:
        virtual_body.stamp_intended("b_stamp", "raise_arm", {"side": "right"})
        virtual_body.stamp_intended("b_stamp", "stop", {})
        virtual_body.note_from_robot(
            "b_stamp",
            {
                "live": {"right_shoulder": 142, "right_elbow": 18, "left_shoulder": 24, "left_elbow": 118},
                "joints": {"right_shoulder": 0, "right_elbow": 5},
                "motion": "idle",
                "pose": "custom",
                "waving": False,
            },
        )
        live = virtual_body.latest_state("b_stamp")
        joints = live.get("joints") or live.get("live") or {}
        self.assertLess(float(joints.get("right_shoulder") or 0), 25, live)
        self.assertFalse(live.get("waving"))

    def test_boot_seed_blocks_html_leftover_raise(self) -> None:
        virtual_body.note_state(
            "b_stamp",
            {
                "mode": "virtual",
                "pose": "custom",
                "motion": "idle",
                "waving": False,
                "joints": {
                    "right_shoulder": 142,
                    "right_elbow": 18,
                    "left_shoulder": 24,
                    "left_elbow": 118,
                },
            },
        )
        virtual_body.seed_boot_pose("b_stamp")
        virtual_body.note_state(
            "b_stamp",
            {
                "mode": "virtual",
                "pose": "custom",
                "joints": {"right_shoulder": 142, "right_elbow": 18, "left_shoulder": 24, "left_elbow": 118},
            },
        )
        live = virtual_body.latest_state("b_stamp")
        joints = live.get("joints") or live.get("live") or {}
        self.assertLess(float(joints.get("right_shoulder") or 0), 25, live)
        self.assertLess(float(joints.get("left_shoulder") or 0), 15, live)
        self.assertFalse(live.get("waving"))


if __name__ == "__main__":
    unittest.main()
