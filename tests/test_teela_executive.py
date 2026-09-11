#!/usr/bin/env python3
"""Qwen-as-executive: one Teela loop, capabilities not lanes, deskd as kernel."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deskd"))

import deskd as d  # noqa: E402
import robot_sim  # noqa: E402


def _tool_names(payload: dict) -> list[str]:
    return [d._openai_tool_name(t) for t in payload.get("tools") or []]


def _families(names: list[str]) -> dict[str, bool]:
    shorts = {n.split("__")[-1] for n in names}
    return {
        "perception": bool(
            shorts
            & {
                "desktop_observe",
                "desktop_state",
                "desktop_screenshot",
                "teela_get_body_state",
                "robot_status",
            }
        ),
        "body": bool(
            shorts
            & {
                "teela_body_action",
                "teela_gesture",
                "teela_stop",
                "robot_pose",
                "robot_joint",
                "robot_motion",
            }
        ),
        "computer": bool(
            shorts
            & {
                "desktop_open_file",
                "desktop_type_text",
                "desktop_browser_navigate",
                "desktop_open_app",
                "desktop_click",
                "read_file",
                "list_dir",
                "web_search",
            }
        ),
        "memory": bool(shorts & {"memory_write", "memory_retrieve"}),
        "collab": bool(shorts & {"list_teammates", "message_teammate"}),
        "system": bool(shorts & {"teela_system_check"}),
    }


class _ExecBot:
    def __init__(self) -> None:
        self.id = "b_teela_exec"
        self.kind = "teela-brain"
        self.model = "qwen38-27b-q5"
        self.robot_state = robot_sim.default_state()
        self.messages: list = []
        self.surface = "preview"
        self.desktop_cursor = {"x": 0, "y": 0}
        self.applied: list = []
        self.workspace = Path(tempfile.mkdtemp())
        (self.workspace / "Desktop").mkdir(exist_ok=True)
        (self.workspace / "Desktop" / "note.txt").write_text("hello desk\n", encoding="utf-8")

    def apply_robot(self, body):
        self.applied.append(dict(body))
        result = robot_sim.apply(self.robot_state, body)
        return result, {"type": "desktop.action", "bot_id": self.id, "action": "robot"}

    def record_local_generation(self, *args, **kwargs):
        return None

    def append_msg(self, role, text, **_kwargs):
        self.messages.append({"role": role, "text": text})

    def finish_prompt_turn(self):
        return None

    def close_chunk(self):
        return None


class TeelaExecutiveTests(unittest.TestCase):
    def test_unified_capability_list_every_turn(self) -> None:
        names = [t["function"]["name"] for t in d.teela_capability_tool_specs()]
        fam = _families(names)
        for key in ("perception", "body", "computer", "memory", "collab", "system"):
            self.assertTrue(fam[key], f"missing {key} tools: {names}")
        bot = _ExecBot()
        payload = d.assemble_teela_executive_payload(bot, "hi")
        wave_payload = d.assemble_teela_executive_payload(bot, "can you wave")
        look_payload = d.assemble_teela_executive_payload(
            bot, "open the picture then point at the person in red"
        )
        for p in (payload, wave_payload, look_payload):
            fam = _families(_tool_names(p))
            for key in ("perception", "body", "computer", "memory", "collab", "system"):
                self.assertTrue(fam[key], f"{key} missing for {p['messages'][-1]['content']!r}")
        hint = d.teela_capability_hint("can you wave")
        self.assertTrue(not hint or "body" in hint.lower())
        sys_text = str((wave_payload.get("messages") or [{}])[0].get("content") or "")
        self.assertIn("I feel", sys_text)
        self.assertIn("web_search", names)
        self.assertIn("web_search", d._TEELA_MINIOS_SYS)
        hint = d.teela_capability_hint("try to do the moonwalk")
        self.assertIn("web_search", hint.lower())

    def test_mixed_turn_observe_then_act_then_speak(self) -> None:
        bot = _ExecBot()
        rounds = [
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "bot_desktop__desktop_observe",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c2",
                                    "function": {
                                        "name": "bot_desktop__teela_gesture",
                                        "arguments": '{"gesture":"point","side":"right"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "That's the person I mean."}}]},
        ]
        seen_tools: list[list[str]] = []

        def fake_complete(payload):
            seen_tools.append(_tool_names(payload))
            return rounds.pop(0)

        user = "Open the picture on your desktop, look at it, then point at the person wearing red."
        line = d.run_teela_executive_turn(bot, user, completer=fake_complete)
        self.assertEqual(line, "That's the person I mean.")
        used = [n.split("__")[-1] for n in (getattr(bot, "_teela_tools_used", None) or [])]
        self.assertIn("desktop_observe", used)
        self.assertIn("teela_gesture", used)
        self.assertTrue(seen_tools)
        fam = _families(seen_tools[0])
        self.assertTrue(fam["perception"] and fam["body"] and fam["computer"])
        loop_src = Path(d.__file__).read_text(encoding="utf-8").split("def _run_prompt_loop", 1)[1].split("def ", 1)[0]
        self.assertIn("run_teela_executive_turn", loop_src)
        self.assertNotIn("teela_turn_lane", loop_src)
        self.assertNotIn('if lane == "minios"', loop_src)
        self.assertNotIn('if lane == "talk"', loop_src)
        self.assertNotIn('if lane == "acp"', loop_src)
        self.assertNotIn("run_teela_minios_turn", loop_src)

    def test_run_prompt_loop_uses_executive_not_lanes(self) -> None:
        bot = _ExecBot()
        bot.status = "Working…"
        bot.control = "local"
        bot._motor_hold = False
        bot._motor_user = ""
        rounds = [
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "bot_desktop__desktop_observe",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c2",
                                    "function": {
                                        "name": "bot_desktop__teela_gesture",
                                        "arguments": '{"gesture":"point","side":"right"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "That's the person I mean."}}]},
        ]

        def fake_complete(payload):
            names = _tool_names(payload)
            fam = _families(names)
            self.assertTrue(fam["perception"] and fam["body"] and fam["computer"])
            return rounds.pop(0)

        bot._teela_completer = fake_complete
        handler = object.__new__(d.Handler)
        executed: list[str] = []
        real_exec = d.execute_teela_allowed_tool

        def spy_exec(b, name, args=None):
            executed.append(d.canonicalize_tool_name(str(name or ""), None).split("__")[-1])
            return real_exec(b, name, args)

        def lane_must_not_run(*_a, **_k):
            raise AssertionError("teela_turn_lane must not decide the Teela turn")

        with patch.object(d, "teela_turn_lane", side_effect=lane_must_not_run):
            with patch.object(d, "execute_teela_allowed_tool", side_effect=spy_exec):
                d.Handler._run_prompt_loop(
                    handler,
                    bot,
                    "Open the picture on your desktop, look at it, then point at the person wearing red.",
                )
        self.assertIn("desktop_observe", executed)
        self.assertIn("teela_gesture", executed)
        spoken = " ".join(str(m.get("text") or "") for m in bot.messages if m.get("role") == "assistant")
        self.assertIn("That's the person I mean.", spoken)

    def test_greeting_speaks_without_tools(self) -> None:
        bot = _ExecBot()

        def fake_complete(payload):
            self.assertTrue(_families(_tool_names(payload))["body"])
            return {"choices": [{"message": {"content": "Hey — I'm here."}}]}

        def lane_must_not_run(*_a, **_k):
            raise AssertionError("teela_turn_lane must not decide the Teela turn")

        with patch.object(d, "teela_turn_lane", side_effect=lane_must_not_run):
            line = d.run_teela_executive_turn(bot, "hi", completer=fake_complete)
        self.assertIn("hey", (line or "").lower())
        used = getattr(bot, "_teela_tools_used", None) or []
        self.assertFalse(used)
        self.assertFalse(bot.applied)

    def test_wave_does_not_replay_the_greeting(self) -> None:
        bot = _ExecBot()
        rounds = [
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "w1",
                                    "function": {
                                        "name": "bot_desktop__robot_pose",
                                        "arguments": '{"pose":"wave"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": "[laugh] Hi there! I'm Teela — nice to meet you. What can I do for you?"
                        }
                    }
                ]
            },
        ]

        def fake_complete(_payload):
            return rounds.pop(0)

        line = d.run_teela_executive_turn(bot, "Can you wave?", completer=fake_complete)
        used = [n.split("__")[-1] for n in (getattr(bot, "_teela_tools_used", None) or [])]
        self.assertTrue(any("wave" in n or n in {"teela_body_action", "teela_gesture", "robot_pose"} for n in used), used)
        self.assertNotIn("nice to meet you", (line or "").lower())
        self.assertTrue(re.search(r"wav|hand|done", (line or "").lower()), line)

    def test_stale_wave_label_does_not_claim_already_waving(self) -> None:
        bot = _ExecBot()
        bot.robot_state["pose"] = "wave"
        bot.robot_state["motion"] = "waving"
        bot.robot_state["waving"] = True
        bot.robot_state["joints"] = dict(robot_sim.POSES["wave"])
        bot.robot_state["live"] = dict(robot_sim.POSES["wave"])
        bot.robot_state["wave_hold"] = dict(robot_sim.POSES["wave"])
        self.assertFalse(robot_sim.observed_waving(bot.robot_state))
        sense = d.proprioception_block(bot.robot_state)
        self.assertNotIn("Wave: yes —", sense)
        self.assertIn("not rocking", sense.lower())

        def fake_complete(_payload):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "I'm already waving at you right now! My hand's up in front of my chest, rocking side to side."
                        }
                    }
                ]
            }

        line = d.run_teela_executive_turn(bot, "Can you wave?", completer=fake_complete)
        used = [n.split("__")[-1] for n in (getattr(bot, "_teela_tools_used", None) or [])]
        self.assertTrue(used)
        self.assertNotIn("already waving", (line or "").lower())
        self.assertNotIn("rocking side to side", (line or "").lower())

    def test_walk_right_is_not_treated_as_validation(self) -> None:
        bot = _ExecBot()
        from teela_cl.attempts import AttemptStore

        AttemptStore(bot.workspace / ".teela").record(
            goal="gesture.wave",
            request="Can you wave?",
            skill="gesture.wave",
            plan=[{"capability": "raise_arm"}],
            awaiting_feedback=True,
        )
        rounds = [
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "w1",
                                    "function": {
                                        "name": "bot_desktop__robot_motion",
                                        "arguments": '{"cmd":"walk","direction":"right"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "Walking right."}}]},
        ]

        def fake_complete(_payload):
            return rounds.pop(0)

        line = d.run_teela_executive_turn(bot, "Can you stop and walk right?", completer=fake_complete)
        self.assertNotIn("validated", (line or "").lower())
        self.assertNotIn("keep that as the", (line or "").lower())
        used = [n.split("__")[-1] for n in (getattr(bot, "_teela_tools_used", None) or [])]
        self.assertTrue(used)

    def test_kernel_rejects_unknown_and_clamps_joints(self) -> None:
        bot = _ExecBot()
        unknown = d.execute_teela_allowed_tool(bot, "search_tool", {"query": "x"})
        self.assertFalse(unknown.get("ok"))
        out = d.execute_teela_allowed_tool(
            bot,
            "bot_desktop__robot_joint",
            {"joint": "right_shoulder", "value": 9999},
        )
        self.assertTrue(out.get("ok"))
        set_rows = out.get("set") or []
        self.assertTrue(set_rows)
        applied = float(set_rows[0]["value"])
        lo, hi = robot_sim.JOINTS["right_shoulder"]
        self.assertGreaterEqual(applied, lo)
        self.assertLessEqual(applied, hi)
        self.assertLess(applied, 9999)
        bad = d.execute_teela_allowed_tool(
            bot,
            "bot_desktop__robot_joint",
            {"joint": "not_a_joint", "value": 10},
        )
        self.assertFalse(bad.get("ok"))
        skill = __import__("virtual_body").submit_action(bot.id, "dance", {})
        self.assertEqual(skill.get("status"), "rejected")

    def test_emergency_stop_skips_completer(self) -> None:
        bot = _ExecBot()

        def boom(_payload):
            raise AssertionError("e-stop must not wait on the model")

        line = d.run_teela_executive_turn(bot, "emergency stop", completer=boom)
        self.assertTrue(line)
        self.assertTrue(bot.robot_state.get("estop") or "stop" in (line or "").lower())
        line2 = d.run_teela_executive_turn(bot, "stop moving", completer=boom)
        self.assertTrue(line2)
        self.assertEqual(getattr(bot, "_teela_effective_mode", ""), "safety")

    def test_web_search_then_attempt_unknown_motion(self) -> None:
        bot = _ExecBot()

        def fake_http(url: str, timeout: float = 8.0):
            if "wikipedia" in url:
                return {
                    "query": {
                        "search": [
                            {
                                "title": "Moonwalk (dance)",
                                "snippet": "a dance that creates the illusion of walking forward while sliding backward",
                            }
                        ]
                    }
                }
            return {
                "Heading": "Moonwalk",
                "AbstractText": "The moonwalk is a dance move that slides backward while appearing to walk forward.",
                "RelatedTopics": [],
            }

        with patch.object(d, "_http_json_get", side_effect=fake_http):
            found = d.execute_teela_allowed_tool(bot, "web_search", {"query": "moonwalk dance"})
        self.assertTrue(found.get("ok"))
        blob = json.dumps(found).lower()
        self.assertIn("backward", blob)
        unknown = d.execute_teela_allowed_tool(bot, "search_tool", {"query": "moonwalk"})
        self.assertFalse(unknown.get("ok"))

        rounds = [
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "s1",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query":"moonwalk dance"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "m1",
                                    "function": {
                                        "name": "bot_desktop__robot_motion",
                                        "arguments": '{"cmd":"walk","direction":"back"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": "I'm approximating a moonwalk — sliding back like a walk going the wrong way."
                        }
                    }
                ]
            },
        ]

        def fake_complete(_payload):
            return rounds.pop(0)

        with patch.object(d, "_http_json_get", side_effect=fake_http):
            line = d.run_teela_executive_turn(
                bot, "try to do the moonwalk", completer=fake_complete
            )
        used = [n.split("__")[-1] for n in (getattr(bot, "_teela_tools_used", None) or [])]
        self.assertIn("web_search", used)
        self.assertIn("robot_motion", used)
        self.assertTrue(re.search(r"body|practiced|skill|moved|desktop", (line or "").lower()))
        self.assertEqual(getattr(bot, "_teela_effective_mode", ""), "mixed")

    def test_moonwalk_kernel_moves_when_model_only_talks(self) -> None:
        bot = _ExecBot()

        def fake_complete(_payload):
            return {
                "choices": [
                    {"message": {"content": "[laugh] Okay, hold on, let's try that!"}}
                ]
            }

        def fake_http(url: str, timeout: float = 8.0):
            return {
                "query": {"search": [{"title": "Moonwalk", "snippet": "slide backward"}]},
                "AbstractText": "walk forward while sliding backward",
                "RelatedTopics": [],
            }

        with patch.object(d, "_http_json_get", side_effect=fake_http):
            line = d.run_teela_executive_turn(
                bot, "try to do the moonwalk", completer=fake_complete
            )
        self.assertTrue(bot.applied, "kernel must move the body even if the model only talks")
        applied = list(getattr(bot, "_teela_applied_caps", []) or [])
        self.assertTrue(applied, "composed plan capabilities must be applied")
        last_plan = getattr(bot, "_teela_last_plan", None) or []
        plan_caps = [str(s.get("capability")) for s in last_plan if isinstance(s, dict)]
        self.assertEqual(applied, plan_caps)
        used = [n.split("__")[-1] for n in (getattr(bot, "_teela_tools_used", None) or [])]
        self.assertTrue(used)
        self.assertTrue(re.search(r"body|practiced|skill|moved", (line or "").lower()))
        pose = str(bot.robot_state.get("pose") or "")
        motion = str(bot.robot_state.get("motion") or "")
        plan = bot.robot_state.get("plan")
        self.assertTrue(pose == "ready" or motion == "walking" or isinstance(plan, dict))
        notes = list((bot.workspace / "Desktop").glob("learn-*.md"))
        self.assertTrue(notes, "practice note must appear on MiniOS Desktop")
        body_md = (bot.workspace / "BODY.md").read_text(encoding="utf-8") if (bot.workspace / "BODY.md").is_file() else ""
        self.assertIn("practice", body_md.lower() + str(notes[0].read_text(encoding="utf-8")).lower())

    def test_existing_capability_executes(self) -> None:
        assessment = d.assess_capability(
            "Can you greet me with your hand?",
            d.state_with_skill("gesture.wave"),
        )
        self.assertEqual(assessment.decision, "execute")
        src = Path(d.__file__).read_text(encoding="utf-8")
        gate = src.split("def teela_needs_learn_attempt", 1)[1].split("def ", 1)[0]
        self.assertNotIn("wave", gate.lower())
        self.assertNotIn("moonwalk", gate.lower())

    def test_unknown_goal_composable_from_existing_primitives(self) -> None:
        from teela_cl.records import InterpretedIntent

        state = d.state_with_skills(
            "balance",
            "weight_shift",
            "foot_slide",
            "heel_raise",
        )
        required = ["balance", "weight_shift", "foot_slide", "heel_raise"]

        def interp(req, skills, model):
            return InterpretedIntent(
                goal="composed_outcome",
                required_capabilities=required,
                domain="embodied",
                expected_outcome=req,
                confidence=0.5,
            )

        assessment = d.assess_capability("Try to moonwalk", state, interpreter=interp)
        self.assertEqual(assessment.decision, "compose")

    def test_missing_capabilities_enters_learning(self) -> None:
        assessment = d.assess_capability(
            "Perform this movement",
            d.state_without_required_skills(),
        )
        self.assertIn(assessment.decision, {"learn", "ask", "refuse", "research"})

    def test_paraphrase_does_not_change_capability_resolution(self) -> None:
        state = d.state_with_skill("gesture.wave")
        requests = [
            "wave",
            "wave at me",
            "say hello with your hand",
            "give me a little greeting gesture",
        ]
        for request in requests:
            assessment = d.assess_capability(request, state)
            self.assertEqual(assessment.decision, "execute", request)

    def test_corrective_feedback_does_not_start_a_new_learn_goal(self) -> None:
        bot = _ExecBot()
        from teela_cl.attempts import AttemptStore

        AttemptStore(bot.workspace / ".teela").record(
            goal="gesture.wave",
            request="wave at me",
            skill="gesture.wave",
            plan=[{"capability": "raise_arm"}, {"capability": "oscillate_wrist"}],
        )

        def boom(_payload):
            raise AssertionError("correction must not start a new model turn")

        line = d.run_teela_executive_turn(bot, "No, that's not right.", completer=boom)
        self.assertTrue(line)
        self.assertTrue(re.search(r"retry|correct|trying|again|like this|got it", (line or "").lower()))
        fb = getattr(bot, "_teela_feedback_result", None)
        self.assertIsNotNone(fb)
        self.assertTrue(fb.handled)
        self.assertNotEqual(getattr(fb, "error_class", ""), "system")

    def test_fast_social_is_one_pass_without_learn_or_rsi(self) -> None:
        from teela_cl.deliberation import choose_turn_policy
        from teela_cl.skill_store import SkillStore

        for req in ("Hi Teela.", "Thanks.", "How are you?"):
            pol = d.choose_turn_policy(req, skills=SkillStore(seed=True).all())
            self.assertEqual(pol.reasoning_mode, "fast", req)
            self.assertFalse(pol.needs_learning, req)
        bot = _ExecBot()
        n = {"c": 0}

        def fake_complete(_payload):
            n["c"] += 1
            return {"choices": [{"message": {"content": "You're welcome."}}]}

        with patch.object(d, "teela_runtime_learn") as learn:
            line = d.run_teela_executive_turn(bot, "Thanks.", completer=fake_complete)
        learn.assert_not_called()
        self.assertLessEqual(n["c"], 1)
        self.assertNotIn("practiced", (line or "").lower())
        src = Path(ROOT / "deskd" / "teela_cl" / "deliberation.py").read_text(encoding="utf-8")
        self.assertNotIn("figure out", src.lower())
        self.assertNotRegex(src.lower(), r'if\s+"wave"')

    def test_fast_known_body_acts_this_turn(self) -> None:
        bot = _ExecBot()
        n = {"c": 0}

        def fake_complete(_payload):
            n["c"] += 1
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Certainly. I understand that you would like me to rotate my head toward the left side of the room now."
                        }
                    }
                ]
            }

        line = d.run_teela_executive_turn(bot, "Turn your head left.", completer=fake_complete)
        used = [x.split("__")[-1] for x in (getattr(bot, "_teela_tools_used", None) or [])]
        self.assertTrue(used, "body tool must run this turn")
        self.assertLessEqual(n["c"], 1)
        self.assertNotIn("certainly. i understand", (line or "").lower())
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", (line or "").strip()) if s]
        self.assertLessEqual(len(sentences), 2)
        pol = getattr(bot, "_teela_turn_policy", None)
        self.assertIsNotNone(pol)
        self.assertEqual(pol.reasoning_mode, "fast")

    def test_deep_minimal_speech_hides_internals(self) -> None:
        from teela_cl.deliberation import TurnFeatures

        bot = _ExecBot()
        bot._teela_turn_features = TurnFeatures(
            confidence=0.2,
            novelty=0.9,
            risk=0.55,
            needs_plan=True,
            needs_tools=True,
            needs_learning=False,
        )
        n = {"c": 0}

        def fake_complete(_payload):
            n["c"] += 1
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "COMPOSE with confidence 0.72. I am entering my LEARNING LOOP. "
                                "I evaluated capability confidence at 0.72 and will now explain every step of the plan in detail for several paragraphs."
                            )
                        }
                    }
                ]
            }

        with patch.object(d, "teela_runtime_learn") as learn:
            line = d.run_teela_executive_turn(
                bot,
                "Watch both objects and pick a safe path around the chair.",
                completer=fake_complete,
            )
        learn.assert_not_called()
        self.assertGreaterEqual(n["c"], 1)
        pol = getattr(bot, "_teela_turn_policy", None)
        self.assertEqual(getattr(pol, "reasoning_mode", ""), "deep")
        self.assertEqual(getattr(pol, "response_mode", ""), "minimal")
        self.assertNotIn("COMPOSE", line or "")
        self.assertNotIn("LEARNING LOOP", (line or "").upper())
        self.assertNotIn("0.72", line or "")
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", (line or "").strip()) if s]
        self.assertLessEqual(len(sentences), 2)

    def test_momentum_binds_short_correction_then_releases(self) -> None:
        from teela_cl.attempts import AttemptStore
        from teela_cl.deliberation import load_momentum

        bot = _ExecBot()
        AttemptStore(bot.workspace / ".teela").record(
            goal="wave at user",
            request="wave at me",
            skill="gesture.wave",
            plan=[{"capability": "raise_arm"}],
            awaiting_feedback=True,
        )
        line = d.run_teela_executive_turn(bot, "Higher.", completer=lambda _p: (_ for _ in ()).throw(AssertionError("local correction")))
        fb = getattr(bot, "_teela_feedback_result", None)
        self.assertIsNotNone(fb)
        self.assertTrue(fb.handled)
        self.assertIn(fb.error_class, {"skill", "execution", "planning"})
        ctx = load_momentum(bot.workspace / ".teela")
        self.assertEqual(ctx.conversation_momentum, "physical_training")
        n = {"c": 0}

        def weather(_p):
            n["c"] += 1
            return {"choices": [{"message": {"content": "I don't have a weather feed."}}]}

        with patch.object(d, "teela_runtime_learn") as learn:
            line2 = d.run_teela_executive_turn(bot, "What's the weather?", completer=weather)
        learn.assert_not_called()
        self.assertGreaterEqual(n["c"], 1)
        self.assertNotIn("validated", (line2 or "").lower())
        self.assertNotIn("practiced", (line2 or "").lower())
        ctx2 = load_momentum(bot.workspace / ".teela")
        self.assertEqual(ctx2.conversation_momentum, "talk")

    def test_one_negative_does_not_launch_rsi(self) -> None:
        from teela_cl.attempts import AttemptStore
        from teela_cl.skill_store import SkillStore

        bot = _ExecBot()
        AttemptStore(bot.workspace / ".teela").record(
            goal="wave at user",
            skill="gesture.wave",
            plan=[{"capability": "raise_arm"}],
        )
        line = d.run_teela_executive_turn(
            bot,
            "No, that's not right.",
            completer=lambda _p: (_ for _ in ()).throw(AssertionError("no model")),
        )
        self.assertTrue(line)
        store = SkillStore(bot.workspace / ".teela", seed=True)
        self.assertFalse([s for s in store.all() if s.source == "learned" or ".v" in s.skill_id])
        kinds = list(getattr(bot, "_teela_events", None) or [])
        self.assertNotIn("improvement_observed", kinds)
        fb = getattr(bot, "_teela_feedback_result", None)
        self.assertFalse(getattr(fb, "rsi_escalated", False))

    def test_resolver_learn_runs_without_try_prefix(self) -> None:
        bot = _ExecBot()

        def fake_complete(_payload):
            return {"choices": [{"message": {"content": "Okay!"}}]}

        def fake_http(_url: str, timeout: float = 8.0):
            return {
                "query": {"search": [{"title": "Dance", "snippet": "a movement"}]},
                "AbstractText": "a movement",
                "RelatedTopics": [],
            }

        with patch.object(d, "_http_json_get", side_effect=fake_http):
            d.run_teela_executive_turn(bot, "moonwalk", completer=fake_complete)
        a = getattr(bot, "_teela_assessment", None)
        self.assertIsNotNone(a)
        self.assertEqual(a.decision, "learn")
        self.assertTrue(bot.applied or getattr(bot, "_teela_applied_caps", None))

    def test_thats_right_validates_awaiting_attempt(self) -> None:
        from teela_cl.attempts import AttemptStore
        from teela_cl.feedback import interpret_feedback
        from teela_cl.self_model import snapshot_self_model

        bot = _ExecBot()
        rec = AttemptStore(bot.workspace / ".teela").record(
            goal="wave at user",
            skill="gesture.wave",
            plan=[{"capability": "raise_arm"}],
            awaiting_feedback=True,
        )
        model = snapshot_self_model()
        for req in ("that's right", "yes that's right", "That's right."):
            fb = interpret_feedback(req, rec, model)
            self.assertEqual(fb.feedback_type, "positive", req)
            self.assertEqual(fb.target_attempt_id, rec.attempt_id)
        n = {"c": 0}

        def boom(_p):
            n["c"] += 1
            raise AssertionError("validation must not start a new model turn")

        line = d.run_teela_executive_turn(bot, "that's right", completer=boom)
        self.assertEqual(n["c"], 0)
        fb = getattr(bot, "_teela_feedback_result", None)
        self.assertIsNotNone(fb)
        self.assertEqual(fb.error_class, "validation")
        self.assertNotIn("like this", (line or "").lower())

    def test_social_after_attempt_is_not_a_correction(self) -> None:
        from teela_cl.attempts import AttemptStore
        from teela_cl.feedback import interpret_feedback
        from teela_cl.self_model import snapshot_self_model

        bot = _ExecBot()
        rec = AttemptStore(bot.workspace / ".teela").record(
            goal="wave at user",
            skill="gesture.wave",
            plan=[{"capability": "raise_arm"}],
            awaiting_feedback=True,
        )
        model = snapshot_self_model()
        for req in ("Hi Teela.", "Thanks."):
            fb = interpret_feedback(req, rec, model)
            self.assertEqual(fb.feedback_type, "none", req)
        n = {"c": 0}

        def fake(_p):
            n["c"] += 1
            return {"choices": [{"message": {"content": "Hey — I'm here."}}]}

        with patch.object(d, "teela_runtime_learn") as learn:
            line = d.run_teela_executive_turn(bot, "Hi Teela.", completer=fake)
        learn.assert_not_called()
        self.assertGreaterEqual(n["c"], 1)
        self.assertNotIn("like this", (line or "").lower())
        pol = getattr(bot, "_teela_turn_policy", None)
        self.assertEqual(getattr(pol, "reasoning_mode", ""), "fast")

    def test_explain_does_not_run_learn_goal(self) -> None:
        n = {"c": 0}

        def fake(_p):
            n["c"] += 1
            return {"choices": [{"message": {"content": "Because the left side was blocked."}}]}

        bot = _ExecBot()
        with patch.object(d, "teela_runtime_learn") as learn:
            line = d.run_teela_executive_turn(bot, "can you explain?", completer=fake)
        learn.assert_not_called()
        self.assertGreaterEqual(n["c"], 1)
        self.assertNotIn("moved my body", (line or "").lower())
        self.assertNotIn("practiced", (line or "").lower())
        pol = getattr(bot, "_teela_turn_policy", None)
        self.assertFalse(getattr(pol, "needs_learning", True))

    def test_greeting_does_not_move_the_body(self) -> None:
        bot = _ExecBot()
        n = {"c": 0}

        def fake(_p):
            n["c"] += 1
            return {"choices": [{"message": {"content": "Hey — I'm here."}}]}

        with patch.object(d, "teela_runtime_learn") as learn:
            line = d.run_teela_executive_turn(bot, "Hi.", completer=fake)
        learn.assert_not_called()
        used = [name.split("__")[-1] for name in (getattr(bot, "_teela_tools_used", None) or [])]
        self.assertFalse(set(used) & {"robot_pose", "robot_motion", "teela_body_action", "robot_joint"})
        self.assertGreaterEqual(n["c"], 1)
        self.assertNotIn("practiced", (line or "").lower())
        self.assertNotIn("like this", (line or "").lower())

    def test_state_observation_does_not_move_or_learn(self) -> None:
        bot = _ExecBot()
        n = {"c": 0}

        def fake(_p):
            n["c"] += 1
            return {"choices": [{"message": {"content": "Yes — my right hand is in a wave."}}]}

        with patch.object(d, "teela_runtime_learn") as learn:
            line = d.run_teela_executive_turn(bot, "You are currently waving", completer=fake)
        learn.assert_not_called()
        used = [name.split("__")[-1] for name in (getattr(bot, "_teela_tools_used", None) or [])]
        self.assertFalse(set(used) & {"robot_pose", "robot_motion", "teela_body_action", "robot_joint", "teela_stop"})
        self.assertGreaterEqual(n["c"], 1)
        self.assertNotIn("practiced", (line or "").lower())
        self.assertNotIn("like this", (line or "").lower())

    def test_put_leg_down_uses_home_not_generic_walk(self) -> None:
        bot = _ExecBot()

        def fake(_p):
            return {"choices": [{"message": {"content": "Okay."}}]}

        with patch.object(d, "teela_runtime_learn") as learn:
            line = d.run_teela_executive_turn(bot, "You can put your leg down", completer=fake)
        learn.assert_not_called()
        used = [name.split("__")[-1] for name in (getattr(bot, "_teela_tools_used", None) or [])]
        self.assertIn("robot_pose", used)
        self.assertNotIn("robot_motion", used)
        self.assertNotIn("practiced", (line or "").lower())
        self.assertNotEqual(bot.robot_state.get("pose"), "walk-cycle")

    def test_no_comma_stop_intercepts_before_model(self) -> None:
        bot = _ExecBot()

        def boom(_p):
            raise AssertionError("stop must not wait on the model")

        line = d.run_teela_executive_turn(bot, "No, stop.", completer=boom)
        self.assertIn("stop", (line or "").lower())
        self.assertEqual(getattr(bot, "_teela_effective_mode", ""), "safety")

    def test_unseen_retry_is_feedback_not_new_learn(self) -> None:
        from teela_cl.attempts import AttemptStore

        bot = _ExecBot()
        AttemptStore(bot.workspace / ".teela").record(
            goal="gesture.wave",
            request="Can you wave?",
            skill="gesture.wave",
            plan=[{"capability": "raise_arm"}],
            awaiting_feedback=True,
        )
        with patch.object(d, "teela_runtime_learn") as learn:
            line = d.run_teela_executive_turn(
                bot,
                "I didn't see you try it, can you try it again",
                completer=lambda _p: (_ for _ in ()).throw(AssertionError("retry is local feedback")),
            )
        learn.assert_not_called()
        fb = getattr(bot, "_teela_feedback_result", None)
        self.assertIsNotNone(fb)
        self.assertTrue(fb.handled)
        self.assertNotIn("practiced the outcome", (line or "").lower())


if __name__ == "__main__":
    unittest.main()
