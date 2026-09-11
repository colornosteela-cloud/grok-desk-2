#!/usr/bin/env python3
"""Fresh-consumer import of shipped deliberation helpers (not unittest)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deskd"))

from teela_cl.attempts import AttemptStore
from teela_cl.deliberation import TurnFeatures, choose_turn_policy
from teela_cl.feedback import interpret_feedback
from teela_cl.self_model import snapshot_self_model
from teela_cl.skill_store import SkillStore


def main() -> int:
    skills = SkillStore(seed=True).all()
    greet = choose_turn_policy("Hi Teela.", skills=skills)
    print("FAST_MODE", greet.reasoning_mode)
    assert greet.reasoning_mode == "fast", greet
    assert not greet.needs_learning

    deep = choose_turn_policy(
        "Watch both objects and pick a safe path around the chair.",
        features=TurnFeatures(
            confidence=0.2,
            novelty=0.9,
            risk=0.55,
            needs_plan=True,
            needs_tools=True,
            needs_learning=False,
        ),
        skills=skills,
    )
    print("DEEP_MODE", deep.reasoning_mode)
    print("DEEP_SPEECH", deep.response_mode)
    assert deep.reasoning_mode == "deep"
    assert deep.response_mode == "minimal"

    tmp = Path(tempfile.mkdtemp())
    store = AttemptStore(tmp)
    rec = store.record(
        goal="wave at user",
        request="wave at me",
        skill="gesture.wave",
        plan=[{"capability": "raise_arm"}],
    )
    fb = interpret_feedback("Higher.", rec, snapshot_self_model())
    print("CORRECTION_TYPE", fb.feedback_type)
    print("CORRECTION_ATTEMPT", fb.target_attempt_id)
    assert fb.feedback_type in {"correction", "negative"}
    assert fb.target_attempt_id == rec.attempt_id
    walk = interpret_feedback("Can you stop and walk right?", rec, snapshot_self_model())
    print("WALK_RIGHT", walk.feedback_type)
    assert walk.feedback_type == "none"
    yes = interpret_feedback("that's right", rec, snapshot_self_model())
    print("THATS_RIGHT", yes.feedback_type)
    assert yes.feedback_type == "positive"
    hi = interpret_feedback("Hi Teela.", rec, snapshot_self_model())
    print("HI_AFTER_ATTEMPT", hi.feedback_type)
    assert hi.feedback_type == "none"
    print("CONSUMER_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
