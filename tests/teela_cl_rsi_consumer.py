#!/usr/bin/env python3
"""Fresh-consumer import of shipped Teela CL/RSI entry points (not unittest)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deskd"))

from teela_cl.learning import LearnContext, compose_plan, learn_goal
from teela_cl.memory_kinds import TypedMemory
from teela_cl.resolver import assess_capability
from teela_cl.rsi import RSIPipeline
from teela_cl.self_model import snapshot_self_model
from teela_cl.skill_store import SkillStore
from teela_cl.records import CandidateState, InterpretedIntent


def main() -> int:
    store = SkillStore(seed=True).only(["gesture.wave"])
    greet = assess_capability("greet me with your hand", skills=store)
    print("GREET_DECISION", greet.decision)
    print("GREET_SKILL", ",".join(greet.matched_skills))
    assert greet.decision == "execute", greet

    prims = SkillStore(seed=True).only(["balance", "weight_shift", "foot_slide", "heel_raise"])
    required = ["balance", "weight_shift", "foot_slide", "heel_raise"]

    def interp(req, skills, model):
        return InterpretedIntent(
            goal="composed_outcome",
            required_capabilities=required,
            domain="embodied",
            expected_outcome=req,
            confidence=0.5,
        )

    unknown = assess_capability("Try to moonwalk", skills=prims, interpreter=interp)
    print("COMPOSE_DECISION", unknown.decision)
    plan = compose_plan(unknown.goal, unknown.required_capabilities)
    print("COMPOSE_PLAN_LEN", len(plan))
    print("COMPOSE_CAPS", ",".join(unknown.required_capabilities))
    assert unknown.decision in {"compose", "practice"} and plan
    assert unknown.required_capabilities == required

    tmp = Path(tempfile.mkdtemp())
    learned_store = SkillStore(tmp, seed=False)
    mem = TypedMemory(tmp)
    box = {"n": 0}

    def observer():
        return {"goal_met": box["n"] > 0, "artifact": "ok" if box["n"] else None}

    def executor(_p):
        box["n"] += 1
        return {"ok": True, "ran": True}

    def software_unknown(req, skills, model):
        return InterpretedIntent(
            goal="unknown_outcome",
            required_capabilities=[],
            domain="software",
            expected_outcome=req,
            confidence=0.25,
            notes="No matching software procedure; research or ask.",
        )

    ctx = LearnContext(
        skills=learned_store,
        memory=mem,
        self_model=snapshot_self_model(),
        executor=executor,
        observer=observer,
        max_attempts=3,
        interpreter=software_unknown,
    )
    result = learn_goal("Can you florb the zed protocol", ctx)
    print("LEARN_SUCCESS", result.success)
    assert result.success and result.skill
    reloaded = SkillStore(tmp, seed=False)
    again = assess_capability("Can you florb the zed protocol", skills=reloaded)
    print("RELOAD_DECISION", again.decision)
    assert again.decision == "execute"

    pipe = RSIPipeline(tmp)
    pipe.active.max_attempts = 1
    active_before = json.dumps(pipe.active.to_dict(), sort_keys=True)
    obs = pipe.observe_weakness(
        subsystem="learning",
        problem="too many replays",
        evidence=["repeat"],
        frequency=6,
        hypothesis="raise replay_penalty",
        proposed_improvement="replay_penalty=2",
        expected_benefit="fewer repeats",
        risk="low",
        measurable_success_criteria="replays down",
    )
    cand, rec = pipe.build_candidate(obs, {"max_attempts": 6})
    rec = pipe.evaluate_candidate(cand, rec)
    print("CANDIDATE_STATE", rec.decision)
    assert rec.decision == CandidateState.AWAITING_PROMOTION.value
    assert json.dumps(pipe.active.to_dict(), sort_keys=True) == active_before

    bad, rec_bad = pipe.build_candidate(obs, {"invariants": []})
    rec_bad = pipe.evaluate_candidate(
        bad,
        rec_bad,
        baseline_metrics={"x": 1.0},
        candidate_metrics={"x": 2.0},
        existing_capability_ok=True,
    )
    print("REGRESSION_OR_INVARIANT", rec_bad.state)
    assert rec_bad.state in {CandidateState.FAILED.value, CandidateState.REGRESSED.value}
    print("CONSUMER_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
