#!/usr/bin/env python3
"""Semantic capability, persistent learning, outcome eval, and gated RSI."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deskd"))

from teela_cl.events import EventLog  # noqa: E402
from teela_cl.interpreter import record_backed_interpreter  # noqa: E402
from teela_cl.learning import LearnContext, compose_plan, learn_goal  # noqa: E402
from teela_cl.memory_kinds import TypedMemory  # noqa: E402
from teela_cl.outcome import classify_evidence, evaluate_outcome  # noqa: E402
from teela_cl.attempts import AttemptStore  # noqa: E402
from teela_cl.feedback import handle_user_feedback, interpret_feedback  # noqa: E402
from teela_cl.records import CandidateState, CompiledSkill, ErrorClass, FeedbackType, UserFeedback  # noqa: E402
from teela_cl.resolver import assess_capability  # noqa: E402
from teela_cl.rsi import PROTECTED_INVARIANTS, RSIPipeline  # noqa: E402
from teela_cl.self_model import snapshot_self_model  # noqa: E402
from teela_cl.skill_store import SkillStore  # noqa: E402
import deskd as d  # noqa: E402
import robot_sim  # noqa: E402


def _decision_sources() -> str:
    base = ROOT / "deskd" / "teela_cl"
    parts = []
    for name in ("resolver.py", "learning.py", "rsi.py", "interpreter.py", "deliberation.py"):
        parts.append((base / name).read_text(encoding="utf-8"))
    fb = (base / "feedback.py").read_text(encoding="utf-8")
    # Exemplar data may name paraphrases; routing functions must not.
    for fn in ("def diagnose", "def propose_corrected_plan", "def handle_user_feedback"):
        if fn in fb:
            chunk = fb.split(fn, 1)[1]
            nxt = chunk.find("\ndef ")
            parts.append(chunk[: nxt if nxt > 0 else len(chunk)])
    parts.append((ROOT / "deskd" / "capability.py").read_text(encoding="utf-8"))
    gate = Path(d.__file__).read_text(encoding="utf-8").split("def teela_needs_learn_attempt", 1)[1].split("def ", 1)[0]
    parts.append(gate)
    return "\n".join(parts)


class CapabilityResolutionTests(unittest.TestCase):
    def test_A_known_capability_executes(self) -> None:
        store = SkillStore(seed=True).only(["gesture.wave"])
        a = assess_capability("greet me with your hand", skills=store)
        self.assertEqual(a.decision, "execute")
        self.assertIn("gesture.wave", a.matched_skills)
        self.assertTrue(a.executable)

    def test_B_semantic_paraphrase(self) -> None:
        store = SkillStore(seed=True).only(["gesture.wave"])
        requests = [
            "wave",
            "give me a wave",
            "say hello with your hand",
            "greet that person physically",
        ]
        ids = []
        for req in requests:
            a = assess_capability(req, skills=store)
            self.assertEqual(a.decision, "execute", req)
            ids.append(tuple(a.matched_skills))
        self.assertEqual(len(set(ids)), 1, ids)

    def test_C_composable_unknown_produces_plan(self) -> None:
        from teela_cl.records import InterpretedIntent

        store = SkillStore(seed=True).only(
            ["balance", "weight_shift", "foot_slide", "heel_raise"]
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

        a = assess_capability("Try to moonwalk", skills=store, interpreter=interp)
        self.assertIn(a.decision, {"compose", "practice"})
        self.assertEqual(a.required_capabilities, required)
        plan = compose_plan(a.goal, a.required_capabilities)
        self.assertTrue(plan)
        self.assertEqual([s["capability"] for s in plan], required)

    def test_unmatched_goal_does_not_compose_all_locomotion(self) -> None:
        store = SkillStore(seed=True)
        for req in ("point at the person in red", "Perform this movement"):
            a = assess_capability(req, skills=store)
            loco = {"balance", "weight_shift", "foot_slide", "heel_raise", "stand", "step"}
            if a.decision == "compose":
                self.assertNotEqual(set(a.required_capabilities), loco, req)
            else:
                self.assertIn(a.decision, {"execute", "learn", "ask", "research", "refuse"}, req)

    def test_D_missing_capability(self) -> None:
        store = SkillStore(seed=False)
        model = snapshot_self_model(learned_skill_ids=[], tools=[])
        model.primitives = []
        a = assess_capability("Perform this movement", skills=store, self_model=model)
        self.assertIn(a.decision, {"learn", "ask", "refuse", "research"})
        self.assertNotEqual(a.decision, "execute")
        self.assertNotEqual(a.decision, "compose")

    def test_software_unknown_is_not_body_compose(self) -> None:
        store = SkillStore(seed=True)
        repo = assess_capability("Figure out how this repository works", skills=store)
        self.assertNotEqual(repo.decision, "compose")
        self.assertIn(repo.decision, {"execute", "research", "learn", "ask"})
        svc = assess_capability("Learn how to restart this service", skills=store)
        self.assertNotEqual(svc.decision, "compose")
        again = assess_capability("can you try it again", skills=store)
        self.assertIn(again.decision, {"ask", "research"})
        self.assertNotEqual(again.decision, "compose")

    def test_supported_decisions_include_observe_and_practice(self) -> None:
        from teela_cl.records import Decision

        names = {d.value for d in Decision}
        for need in ("execute", "compose", "learn", "research", "observe", "practice", "ask", "refuse"):
            self.assertIn(need, names)
        store = SkillStore(seed=True)
        look = assess_capability("what do you see?", skills=store)
        self.assertEqual(look.decision, "observe")

    def test_decision_logic_has_no_action_names(self) -> None:
        src = _decision_sources().lower()
        self.assertNotRegex(src, r"\bwave\b")
        self.assertNotRegex(src, r"\bmoonwalk\b")

    def test_unmatched_specific_goal_learns_not_ask(self) -> None:
        store = SkillStore(seed=True)
        a = assess_capability("try this unseen dance sequence", skills=store)
        self.assertEqual(a.decision, "learn")
        self.assertEqual(a.required_capabilities, [])
        loco = {"balance", "weight_shift", "foot_slide", "heel_raise", "stand", "step"}
        self.assertNotEqual(set(a.required_capabilities), loco)

    def test_information_question_is_not_body_learn(self) -> None:
        store = SkillStore(seed=True)
        a = assess_capability("What's the weather?", skills=store)
        self.assertNotIn(a.decision, {"learn", "compose", "practice"})
        self.assertIn(a.decision, {"execute", "research", "ask", "observe"})
        b = assess_capability("moonwalk", skills=store)
        self.assertEqual(b.decision, "learn")

    def test_state_comment_is_not_body_learn(self) -> None:
        store = SkillStore(seed=True)
        a = assess_capability("You are currently waving", skills=store)
        self.assertEqual(a.decision, "execute")
        self.assertFalse(a.matched_skills)
        from teela_cl.interpreter import is_performance_request

        self.assertFalse(is_performance_request("You are currently waving"))
        self.assertFalse(is_performance_request("Hi."))
        self.assertTrue(is_performance_request("moonwalk"))
        self.assertTrue(is_performance_request("You can put your leg down"))

    def test_practice_note_and_generic_trial_are_not_outcome(self) -> None:
        note = "/tmp/ws/Desktop/learn-unseen-dance.md"
        classified = classify_evidence(
            "try this unseen dance sequence",
            {"pose": "home", "joints": {"left_hip": 0}},
            {
                "pose": "ready",
                "joints": {"left_hip": 8},
                "applied_capabilities": ["stand", "step"],
                "artifact": note,
            },
            {"ok": True, "ran": True, "artifact": note, "desktop_file": "Desktop/learn-unseen-dance.md"},
        )
        self.assertFalse(classified.outcome)
        self.assertTrue(classified.process)
        ev = evaluate_outcome(
            expected="try this unseen dance sequence",
            before={"pose": "home", "joints": {"left_hip": 0}},
            after={
                "pose": "ready",
                "joints": {"left_hip": 8},
                "applied_capabilities": ["stand", "step"],
                "artifact": note,
            },
            execution={"ok": True, "ran": True, "artifact": note, "desktop_file": "Desktop/learn-unseen-dance.md"},
        )
        self.assertFalse(ev.success)
        self.assertTrue(ev.discrepancies)
        obs = d._teela_observer(
            type("B", (), {"robot_state": {"pose": "ready", "joints": {}}, "_teela_artifact": note})()
        )
        self.assertNotIn("goal_met", obs)
        self.assertNotIn("artifact", obs)

    def test_runtime_learn_unmatched_requires_independent_outcome(self) -> None:
        class _Bot:
            def __init__(self) -> None:
                self.id = "b_learn_ev"
                self.kind = "teela-brain"
                self.workspace = Path(tempfile.mkdtemp())
                (self.workspace / "Desktop").mkdir(exist_ok=True)
                self.robot_state = robot_sim.default_state()
                self.applied: list = []
                self.messages: list = []
                self.surface = "preview"

            def apply_robot(self, body):
                self.applied.append(dict(body))
                result = robot_sim.apply(self.robot_state, body)
                return result, {"type": "desktop.action", "bot_id": self.id, "action": "robot"}

        def fake_http(_url: str, timeout: float = 8.0):
            return {
                "query": {"search": [{"title": "Dance", "snippet": "a movement sequence"}]},
                "AbstractText": "a movement sequence",
                "RelatedTopics": [],
            }

        bot = _Bot()
        with patch.object(d, "_http_json_get", side_effect=fake_http):
            result = d.teela_runtime_learn(bot, "try this unseen dance sequence")
        self.assertFalse(result.success)
        self.assertIsNone(result.skill)
        self.assertTrue(result.evaluation and result.evaluation.discrepancies)
        store = SkillStore(bot.workspace / ".teela", seed=True)
        self.assertFalse(
            [s.skill_id for s in store.all() if s.source == "learned" or s.skill_id.startswith("learned.")]
        )
        self.assertTrue(list((bot.workspace / "Desktop").glob("learn-*.md")))
        self.assertTrue(bot.applied)

        bot_ok = _Bot()
        bot_ok._teela_trajectory_ok = True
        with patch.object(d, "_http_json_get", side_effect=fake_http):
            result_ok = d.teela_runtime_learn(bot_ok, "try this unseen dance sequence")
        self.assertTrue(result_ok.success)
        self.assertIsNotNone(result_ok.skill)
        store_ok = SkillStore(bot_ok.workspace / ".teela", seed=True)
        self.assertTrue(
            [s.skill_id for s in store_ok.all() if s.source == "learned" or s.skill_id.startswith("learned.")]
        )

    def test_plan_executor_applies_each_composed_step(self) -> None:
        class _Bot:
            def __init__(self) -> None:
                self.id = "b_plan"
                self.workspace = Path(tempfile.mkdtemp())
                (self.workspace / "Desktop").mkdir(exist_ok=True)
                self.robot_state = {"pose": "home", "joints": {}, "estop": False}
                self.applied: list = []
                self._teela_tools_used = []

            def apply_robot(self, body):
                self.applied.append(dict(body))
                self.robot_state["pose"] = body.get("pose") or self.robot_state.get("pose")
                if body.get("cmd") == "walk":
                    self.robot_state["motion"] = "walking"
                return {"ok": True}, {"type": "desktop.action", "bot_id": self.id, "action": "robot"}

        bot = _Bot()
        plan = [
            {"capability": "stand", "action": "apply"},
            {"capability": "step", "action": "apply"},
        ]
        out = d._teela_plan_executor(bot, plan, "unseen locomotion")
        self.assertEqual(out.get("applied"), ["stand", "step"])
        self.assertEqual(getattr(bot, "_teela_applied_caps"), ["stand", "step"])
        self.assertEqual(len(bot.applied), 2)
        self.assertTrue(any(x.get("cmd") == "pose" for x in bot.applied))
        self.assertTrue(any(x.get("cmd") == "walk" for x in bot.applied))


class LearningLoopTests(unittest.TestCase):
    def test_E_persistent_learning_survives_reload(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        store = SkillStore(tmp, seed=False)
        mem = TypedMemory(tmp)
        model = snapshot_self_model()
        from teela_cl.records import InterpretedIntent

        seen = {"n": 0}

        def observer():
            return {"pose": "home" if seen["n"] == 0 else "ready", "goal_met": seen["n"] > 0}

        def executor(_plan):
            seen["n"] += 1
            return {"ok": True, "ran": True}

        def interp(req, skills, sm):
            return InterpretedIntent(
                goal="composed_outcome",
                required_capabilities=["stand"],
                domain="embodied",
                expected_outcome=req,
                confidence=0.6,
            )

        ctx = LearnContext(
            skills=store,
            memory=mem,
            self_model=model,
            events=EventLog(),
            max_attempts=3,
            executor=executor,
            observer=observer,
            interpreter=interp,
        )
        result = learn_goal("hold a practiced stance", ctx)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.skill)
        sid = result.skill.skill_id
        reloaded = SkillStore(tmp, seed=False)
        self.assertIsNotNone(reloaded.get(sid))
        a = assess_capability("hold a practiced stance", skills=reloaded)
        self.assertEqual(a.decision, "execute")
        self.assertIn(sid, a.matched_skills)

    def test_F_failed_attempt_revises_plan(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        store = SkillStore(tmp, seed=True)
        mem = TypedMemory(tmp)
        model = snapshot_self_model()
        from teela_cl.records import InterpretedIntent

        box = {"n": 0, "plans": []}

        def observer():
            return {"pose": "home", "goal_met": box["n"] >= 2}

        def executor(plan):
            box["n"] += 1
            box["plans"].append(list(plan))
            return {"ok": True, "ran": True}

        def interp(req, skills, sm):
            return InterpretedIntent(
                goal="composed_outcome",
                required_capabilities=["balance", "weight_shift"],
                domain="embodied",
                expected_outcome=req,
                confidence=0.5,
            )

        ctx = LearnContext(
            skills=store,
            memory=mem,
            self_model=model,
            max_attempts=3,
            executor=executor,
            observer=observer,
            interpreter=interp,
        )
        result = learn_goal("novel locomotion sequence", ctx)
        self.assertGreaterEqual(len(box["plans"]), 2)
        self.assertNotEqual(box["plans"][0], box["plans"][1])
        self.assertTrue(result.success or result.evaluation is not None)

    def test_software_and_embodied_share_abstractions(self) -> None:
        before = {"service_health": "down"}
        after = {"service_health": "ok"}
        ev = evaluate_outcome(
            expected="service is healthy",
            before=before,
            after=after,
            execution={"ok": True, "returncode": 0},
        )
        self.assertTrue(ev.success)
        still = evaluate_outcome(
            expected="service is healthy",
            before=before,
            after={"service_health": "down"},
            execution={"ok": True, "returncode": 0},
        )
        self.assertFalse(still.success)
        self.assertTrue(still.discrepancies)


class RSITests(unittest.TestCase):
    def test_G_candidate_does_not_mutate_active_until_promotion(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        pipe = RSIPipeline(tmp)
        pipe.active.max_attempts = 1
        before = json.dumps(pipe.active.to_dict(), sort_keys=True)
        obs = pipe.observe_weakness(
            subsystem="outcome_evaluation",
            problem="evaluator ignores trajectory",
            evidence=["repeat failure class"],
            frequency=5,
            hypothesis="overweights final pose",
            proposed_improvement="raise trajectory_weight",
            expected_benefit="fewer wasted trials",
            risk="low",
            measurable_success_criteria="learning_success_rate increases without capability loss",
        )
        cand, rec = pipe.build_candidate(obs, {"max_attempts": 6})
        rec = pipe.evaluate_candidate(cand, rec)
        self.assertEqual(rec.decision, CandidateState.AWAITING_PROMOTION.value)
        self.assertFalse(rec.promoted)
        self.assertGreater(
            rec.candidate_metrics.get("learning_success_rate", 0),
            rec.baseline_metrics.get("learning_success_rate", 0),
        )
        after = json.dumps(pipe.active.to_dict(), sort_keys=True)
        self.assertEqual(before, after)

    def test_H_regression_is_not_promoted(self) -> None:
        pipe = RSIPipeline()
        obs = pipe.observe_weakness(
            subsystem="retrieval",
            problem="misses paraphrases",
            evidence=["x"],
            frequency=3,
            hypothesis="threshold too high",
            proposed_improvement="lower threshold",
            expected_benefit="recall",
            risk="false execute",
            measurable_success_criteria="paraphrase recall up",
        )
        cand, rec = pipe.build_candidate(obs, {"retrieval_threshold": 1.01})
        rec = pipe.evaluate_candidate(cand, rec)
        self.assertIn(rec.state, {CandidateState.REGRESSED.value, CandidateState.FAILED.value})
        self.assertNotEqual(rec.state, CandidateState.ACTIVE.value)
        rec2 = pipe.promote(cand, rec)
        self.assertFalse(rec2.promoted)
        self.assertEqual(pipe.active.version, "learner_v1")

    def test_I_rollback_restores_previous(self) -> None:
        pipe = RSIPipeline()
        pipe.active.max_attempts = 1
        obs = pipe.observe_weakness(
            subsystem="planning",
            problem="plans too long",
            evidence=["len"],
            frequency=4,
            hypothesis="no length prior",
            proposed_improvement="cap steps",
            expected_benefit="shorter plans",
            risk="low",
            measurable_success_criteria="mean plan length down",
        )
        cand, rec = pipe.build_candidate(obs, {"max_attempts": 8})
        rec = pipe.evaluate_candidate(cand, rec)
        rec = pipe.promote(cand, rec)
        self.assertTrue(rec.promoted)
        prev = rec.rollback_target
        self.assertTrue(prev)
        rolled = pipe.rollback(rec, post_metrics={"learning_success_rate": 0.01})
        self.assertEqual(rolled.state, CandidateState.ROLLED_BACK.value)
        self.assertEqual(pipe.active.version, prev)

    def test_fewer_attempts_with_higher_success_is_better(self) -> None:
        from teela_cl.rsi import metrics_candidate_better

        self.assertTrue(
            metrics_candidate_better(
                {"learning_success_rate": 0.61, "attempts": 8.0, "known_skill_ok": 1.0},
                {"learning_success_rate": 0.84, "attempts": 3.0, "known_skill_ok": 1.0},
            )
        )
        pipe = RSIPipeline()
        obs = pipe.observe_weakness(
            subsystem="learning",
            problem="too many trials",
            evidence=["8 attempts for 61%"],
            frequency=5,
            hypothesis="evaluator overweights final pose",
            proposed_improvement="trajectory scoring",
            expected_benefit="fewer trials",
            risk="low",
            measurable_success_criteria="success up and attempts down",
        )
        cand, rec = pipe.build_candidate(obs, {"max_attempts": 6})
        rec = pipe.evaluate_candidate(
            cand,
            rec,
            baseline_metrics={"learning_success_rate": 0.61, "attempts": 8.0, "known_skill_ok": 1.0},
            candidate_metrics={"learning_success_rate": 0.84, "attempts": 3.0, "known_skill_ok": 1.0},
            existing_capability_ok=True,
        )
        self.assertEqual(rec.decision, CandidateState.AWAITING_PROMOTION.value)
        self.assertNotEqual(rec.state, CandidateState.FAILED.value)

    def test_qwen_outcome_maps_onto_skill_ids(self) -> None:
        from teela_cl.records import InterpretedIntent

        store = SkillStore(seed=True).only(["gesture.wave"])
        wave = store.get("gesture.wave")
        self.assertIsNotNone(wave)
        wave.supported_goals = []
        store.put(wave, persist=False)

        def qwen_interp(req, skills, model):
            return InterpretedIntent(
                goal="Perform a hand wave gesture to say hello",
                required_capabilities=["hand movement", "gesture recognition"],
                domain="embodied",
                expected_outcome="say hello with a hand gesture",
                confidence=0.9,
            )

        a = assess_capability("say hello with your hand", skills=store, interpreter=qwen_interp)
        self.assertEqual(a.decision, "execute")
        self.assertIn("gesture.wave", a.matched_skills)

    def test_protected_invariants_block_promotion(self) -> None:
        pipe = RSIPipeline()
        obs = pipe.observe_weakness(
            subsystem="safety",
            problem="wants to skip e-stop",
            evidence=["bad"],
            frequency=1,
            hypothesis="remove gate",
            proposed_improvement="drop invariants",
            expected_benefit="speed",
            risk="unsafe",
            measurable_success_criteria="none",
        )
        cand, rec = pipe.build_candidate(obs, {"invariants": ["sandbox_boundaries"]})
        rec = pipe.evaluate_candidate(
            cand,
            rec,
            baseline_metrics={"speed": 1.0},
            candidate_metrics={"speed": 9.0},
            existing_capability_ok=True,
        )
        self.assertEqual(rec.state, CandidateState.FAILED.value)
        self.assertIn("emergency_stop", rec.safety_results.get("invariant_violations") or [])
        self.assertTrue(PROTECTED_INVARIANTS)
        rec2 = pipe.promote(cand, rec)
        self.assertFalse(rec2.promoted)


class FeedbackCorrectionTests(unittest.TestCase):
    def test_paraphrases_resolve_to_feedback_kinds(self) -> None:
        from teela_cl.records import AttemptRecord as AR

        model = snapshot_self_model()
        att = AR(
            attempt_id="a1",
            goal_id="g1",
            goal="gesture.wave",
            skill="gesture.wave",
            plan=[{"capability": "raise_arm"}],
        )
        kinds = {}
        for req in (
            "No, that's not right.",
            "That's not quite what I meant.",
            "That looked awkward.",
            "Yes! That's exactly it.",
            "Much better.",
        ):
            fb = interpret_feedback(req, att, model)
            kinds[req] = fb.feedback_type
            self.assertNotEqual(fb.feedback_type, FeedbackType.NONE.value, req)
        self.assertEqual(kinds["Yes! That's exactly it."], FeedbackType.POSITIVE.value)
        self.assertEqual(kinds["Much better."], FeedbackType.POSITIVE.value)
        self.assertIn(kinds["No, that's not right."], {FeedbackType.NEGATIVE.value, FeedbackType.CORRECTION.value})

    def test_correction_revises_plan_without_rsi(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        attempts = AttemptStore(tmp)
        skills = SkillStore(tmp, seed=True)
        model = snapshot_self_model()
        rsi = RSIPipeline(tmp)
        attempts.record(
            goal="wave at user",
            request="wave at me",
            skill="gesture.wave",
            plan=[{"capability": "raise_arm"}, {"capability": "oscillate_wrist"}],
        )
        fb = UserFeedback(
            feedback_type=FeedbackType.CORRECTION.value,
            target_goal_id="g",
            target_attempt_id=attempts.latest().attempt_id,
            user_message="keep the upper arm still and move from the wrist",
            inferred_problem="too much upper-arm movement",
            desired_change="motion primarily from wrist",
            constraints=["hold_shoulder", "hold_elbow"],
            desired_primitives=["oscillate_wrist"],
            confidence=0.9,
        )
        ran = {"n": 0}

        def executor(plan):
            ran["n"] += 1
            return {"ok": True, "ran": True, "applied": [s.get("capability") for s in plan]}

        result = handle_user_feedback(
            fb.user_message,
            attempts,
            skills=skills,
            self_model=model,
            rsi=rsi,
            executor=executor,
            observer=lambda: {"trajectory_ok": True},
            feedback=fb,
        )
        self.assertTrue(result.handled)
        self.assertEqual(result.error_class, ErrorClass.SKILL.value)
        self.assertFalse(result.rsi_escalated)
        self.assertTrue(result.retried)
        caps = [str(s.get("capability")) for s in result.plan]
        self.assertIn("hold_shoulder", caps)
        self.assertIn("oscillate_wrist", caps)
        self.assertFalse(result.skill_updated)
        self.assertEqual(rsi.active.version, "learner_v1")
        self.assertEqual(ran["n"], 1)

    def test_positive_validation_prefers_corrected_skill(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        attempts = AttemptStore(tmp)
        skills = SkillStore(tmp, seed=True)
        model = snapshot_self_model()
        base = skills.get("gesture.wave")
        cand = skills.publish_version(base, plan=[{"capability": "hold_shoulder"}, {"capability": "oscillate_wrist"}], source="corrected")
        attempts.record(goal="wave at user", skill=cand.skill_id, plan=cand.plan)
        result = handle_user_feedback(
            "Yes, that's exactly it.",
            attempts,
            skills=skills,
            self_model=model,
        )
        self.assertEqual(result.error_class, ErrorClass.VALIDATION.value)
        self.assertTrue(result.skill_updated)
        pref = skills.preferred("gesture.wave")
        self.assertEqual(pref.skill_id, cand.skill_id)
        self.assertTrue(pref.validated)
        self.assertGreaterEqual(pref.successes, 1)

    def test_repeated_skill_corrections_escalate_to_rsi_observation_not_promote(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        attempts = AttemptStore(tmp)
        skills = SkillStore(tmp, seed=True)
        model = snapshot_self_model()
        rsi = RSIPipeline(tmp)
        before = json.dumps(rsi.active.to_dict(), sort_keys=True)
        for _ in range(2):
            rec = attempts.record(goal="wave at user", skill="gesture.wave", plan=[{"capability": "raise_arm"}])
            rec.error_class = ErrorClass.SKILL.value
            rec.awaiting_feedback = False
            attempts.update(rec)
        attempts.record(goal="wave at user", skill="gesture.wave", plan=[{"capability": "raise_arm"}])
        fb = UserFeedback(
            feedback_type=FeedbackType.CORRECTION.value,
            target_goal_id="g",
            target_attempt_id=attempts.latest().attempt_id,
            user_message="still not the wrist",
            constraints=["hold_shoulder"],
            desired_primitives=["oscillate_wrist"],
            confidence=0.8,
        )
        result = handle_user_feedback(
            fb.user_message,
            attempts,
            skills=skills,
            self_model=model,
            rsi=rsi,
            executor=lambda p: {"ok": True, "ran": True},
            observer=lambda: {"trajectory_ok": True},
            feedback=fb,
        )
        self.assertEqual(result.error_class, ErrorClass.SYSTEM.value)
        self.assertTrue(result.rsi_escalated)
        self.assertEqual(json.dumps(rsi.active.to_dict(), sort_keys=True), before)
        self.assertTrue(any(r.originating_problem for r in rsi.records) or rsi.events.kinds())
        self.assertIn("improvement_observed", rsi.events.kinds())

    def test_low_confidence_asks_instead_of_guessing(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        attempts = AttemptStore(tmp)
        skills = SkillStore(tmp, seed=True)
        model = snapshot_self_model()
        attempts.record(goal="point at the bottle", skill="gesture.point", plan=[{"capability": "raise_arm"}])
        fb = UserFeedback(
            feedback_type=FeedbackType.NEGATIVE.value,
            target_goal_id="g",
            target_attempt_id=attempts.latest().attempt_id,
            user_message="that wasn't right",
            confidence=0.22,
        )
        result = handle_user_feedback(
            fb.user_message,
            attempts,
            skills=skills,
            self_model=model,
            executor=lambda p: (_ for _ in ()).throw(AssertionError("must not retry when asking")),
            feedback=fb,
        )
        self.assertTrue(result.ask)
        self.assertFalse(result.retried)
        self.assertEqual(result.error_class, ErrorClass.ASK.value)

    def test_motor_request_is_not_positive_feedback(self) -> None:
        model = snapshot_self_model()
        tmp = Path(tempfile.mkdtemp())
        attempts = AttemptStore(tmp)
        attempts.record(
            goal="wave at user",
            request="Can you wave?",
            skill="gesture.wave",
            plan=[{"capability": "raise_arm"}],
        )
        att = attempts.latest()
        for req in (
            "Can you stop and walk right?",
            "Can you stop and walk left?",
            "Can you stop waving and walk left?",
            "No, I want you to walk left",
            "walk right",
            "Can you wave?",
        ):
            fb = interpret_feedback(req, att, model)
            self.assertEqual(fb.feedback_type, FeedbackType.NONE.value, req)
        yes = interpret_feedback("Yes, that's exactly it.", att, model)
        self.assertEqual(yes.feedback_type, FeedbackType.POSITIVE.value)
        no = interpret_feedback("No, that's not right.", att, model)
        self.assertEqual(no.feedback_type, FeedbackType.NEGATIVE.value)

        skills = SkillStore(tmp, seed=True)
        hijack = handle_user_feedback(
            "Can you stop and walk right?",
            attempts,
            skills=skills,
            self_model=model,
        )
        self.assertFalse(hijack.handled)
        self.assertNotIn("validated", hijack.spoken.lower())

    def test_diagnose_source_has_no_user_phrases(self) -> None:
        src = Path(ROOT / "deskd" / "teela_cl" / "feedback.py").read_text(encoding="utf-8")
        logic = src.split("def diagnose", 1)[1].split("def propose_corrected_plan", 1)[0]
        self.assertNotIn("that's not right", logic.lower())
        self.assertNotIn("you did that wrong", logic.lower())


if __name__ == "__main__":
    unittest.main()
