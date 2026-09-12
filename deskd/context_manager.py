"""Situation-aware working-set builder between persistent Teela state and Qwen.

Publishes decisions through working_memory (reason codes, real compact/retrieve
events). Does not own a second facts store or the UI routes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import memory as botmem
import telemetry as tel
import working_memory as wm

CONVERSATION_LIMITS = {
    wm.PRESSURE_GREEN: 12,
    wm.PRESSURE_YELLOW: 8,
    wm.PRESSURE_ORANGE: 6,
    wm.PRESSURE_RED: 4,
    wm.PRESSURE_CRITICAL: 3,
}

PRESSURE_COMPACT_THRESHOLD = {
    wm.PRESSURE_GREEN: botmem.DEFAULT_COMPACT_THRESHOLD,
    wm.PRESSURE_YELLOW: 2500,
    wm.PRESSURE_ORANGE: 1600,
    wm.PRESSURE_RED: 800,
    wm.PRESSURE_CRITICAL: 400,
}

LARGE_TASK_MEMORY = 96000
HEADROOM_FRACTION = 0.35


def occupancy_pressure(bot: Any) -> str:
    used = wm.occupancy_used(bot)
    cap = wm.occupancy_capacity(bot)
    stats = wm.occupancy_stats(used, cap)
    return stats["pressure"] or wm.PRESSURE_GREEN


def conversation_turn_limit(bot: Any) -> int:
    pressure = occupancy_pressure(bot)
    n = CONVERSATION_LIMITS.get(pressure, 8)
    if _large_task_pending(bot) and pressure in {wm.PRESSURE_GREEN, wm.PRESSURE_YELLOW}:
        return max(n, 24)
    return n


def checkpoint_path(bot: Any) -> Path | None:
    mgr = getattr(bot, "memory", None)
    if mgr is None:
        return None
    return Path(mgr.root) / "task_checkpoint.json"


def load_task_checkpoint(bot: Any) -> dict[str, Any] | None:
    path = checkpoint_path(bot)
    if path is None or not path.is_file():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return rec if isinstance(rec, dict) else None


def save_task_checkpoint(bot: Any, fields: dict[str, Any] | None = None) -> dict[str, Any] | None:
    path = checkpoint_path(bot)
    if path is None:
        return None
    rec = dict(load_task_checkpoint(bot) or {})
    if fields:
        rec.update({k: v for k, v in fields.items() if v is not None})
    rec.setdefault("status", "RUNNING")
    rec.setdefault("task_id", rec.get("task_id") or "task_1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    mgr = getattr(bot, "memory", None)
    if mgr is not None:
        mgr.context_events.append(
            {
                "type": "TASK_CHECKPOINT_SAVED",
                "ts": botmem._iso_now(),
                "goal": rec.get("goal"),
                "current_step": rec.get("current_step"),
                "status": rec.get("status"),
            }
        )
    return rec


def complete_task(bot: Any) -> None:
    path = checkpoint_path(bot)
    if path is None:
        return
    rec = dict(load_task_checkpoint(bot) or {})
    if str(rec.get("status") or "").lower() in {"done", "completed"} and rec:
        return
    rec["status"] = "completed"
    rec["workspace"] = ""
    rec["remaining_steps"] = []
    rec["next_action"] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    mgr = getattr(bot, "memory", None)
    if mgr is not None:
        mgr._wm_checkpoint_loaded_fp = _checkpoint_key(rec)  # type: ignore[attr-defined]


def _large_task_pending(bot: Any) -> bool:
    rec = load_task_checkpoint(bot) or {}
    if str(rec.get("status") or "").lower() in {"done", "completed"}:
        return False
    if rec.get("workspace"):
        return True
    steps = rec.get("remaining_steps") or []
    return len(steps) >= 3


def _load_momentum(bot: Any) -> Any:
    try:
        from teela_cl.deliberation import load_momentum

        ws = getattr(bot, "workspace", None)
        root = Path(str(ws)) / ".teela" if ws else None
        return load_momentum(root)
    except Exception:
        return None


def _momentum_is_active_task(ctx: Any) -> bool:
    """True only while Teela is actually in a task/training loop — not leftover goal + talk."""
    if ctx is None:
        return False
    mom = str(getattr(ctx, "conversation_momentum", "") or "").strip().lower()
    if mom in {"physical_training", "task", "working", "agent"}:
        return True
    if bool(getattr(ctx, "user_feedback_expected", False)):
        return True
    return False


def _checkpoint_key(rec: dict[str, Any] | None) -> str:
    if not rec:
        return ""
    ws = str(rec.get("workspace") or "")
    payload = {
        "task_id": rec.get("task_id"),
        "goal": rec.get("goal"),
        "status": rec.get("status"),
        "current_step": rec.get("current_step"),
        "next_action": rec.get("next_action"),
        "remaining_steps": rec.get("remaining_steps") or [],
        "workspace_sha": hashlib.sha256(ws.encode()).hexdigest()[:16] if ws else "",
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _user_continues_task(text: str) -> bool:
    q = (text or "").strip().lower()
    if not q:
        return False
    return q in {"continue", "keep going", "resume"} or q.startswith("continue ")


def _user_explicitly_done(text: str) -> bool:
    q = (text or "").strip().lower()
    if not q:
        return False
    if q in {"done", "that's all", "thats all", "thanks", "thank you", "nevermind", "never mind", "forget it"}:
        return True
    if q.startswith(("that's all", "thats all", "thanks,", "thank you,")):
        return True
    return False


def _user_moved_on(text: str) -> bool:
    q = (text or "").strip().lower()
    if not q:
        return False
    if _user_continues_task(q) or _user_explicitly_done(q):
        return False
    greet = ("hi", "hello", "hey", "howdy", "yo")
    if q in greet or q.rstrip("!.") in greet:
        return True
    if q.startswith(("hi ", "hello ", "hey ", "how are you")):
        return True
    return False


def _sync_checkpoint_from_momentum(bot: Any, user_text: str = "", ctx: Any = None) -> None:
    """Refresh step/next_action only for an already-RUNNING same-goal checkpoint.

    Leftover momentum must never create or revive a checkpoint. New long-horizon
    work starts via explicit save_task_checkpoint, not planted physical_training.
    Continue/resume is handled in before_assemble against that live file.
    """
    if ctx is None:
        ctx = _load_momentum(bot)
    if not _momentum_is_active_task(ctx):
        return
    goal = str(getattr(ctx, "goal", "") or "").strip()
    if not goal:
        return
    rec = load_task_checkpoint(bot)
    rec_status = str((rec or {}).get("status") or "").lower()
    rec_goal = str((rec or {}).get("goal") or "").strip().lower()
    same_live = bool(rec) and rec_status not in {"done", "completed"} and rec_goal == goal.lower()
    if not same_live:
        return
    save_task_checkpoint(
        bot,
        {
            "current_step": getattr(ctx, "active_context", None) or None,
            "next_action": getattr(ctx, "active_context", None) or None,
            "status": "RUNNING",
        },
    )


def maybe_complete_task(bot: Any, user_text: str = "", ctx: Any = None) -> bool:
    """Complete a RUNNING checkpoint on the live assemble path when work is over or the user moved on."""
    rec = load_task_checkpoint(bot)
    if not rec or str(rec.get("status") or "").lower() in {"done", "completed"}:
        return False
    if _user_continues_task(user_text):
        return False
    if _user_explicitly_done(user_text):
        complete_task(bot)
        return True
    if ctx is None:
        ctx = _load_momentum(bot)
    long_horizon = bool(rec.get("remaining_steps") or rec.get("workspace"))
    if long_horizon:
        return False
    if not _momentum_is_active_task(ctx) or _user_moved_on(user_text):
        complete_task(bot)
        return True
    return False


def render_checkpoint(rec: dict[str, Any]) -> str:
    if not rec or str(rec.get("status") or "").lower() in {"done", "completed"}:
        return ""
    lines = ["# Active task checkpoint"]
    if rec.get("goal"):
        lines.append(f"Goal: {rec['goal']}")
    if rec.get("status"):
        lines.append(f"Status: {rec['status']}")
    if rec.get("current_step"):
        lines.append(f"Current step: {rec['current_step']}")
    completed = rec.get("completed_steps") or []
    if completed:
        lines.append("Completed: " + "; ".join(str(x) for x in completed[:12]))
    remaining = rec.get("remaining_steps") or []
    if remaining:
        lines.append("Remaining: " + "; ".join(str(x) for x in remaining[:12]))
    if rec.get("next_action"):
        lines.append(f"Next: {rec['next_action']}")
    if rec.get("errors"):
        lines.append("Errors: " + "; ".join(str(x) for x in rec["errors"][:6]))
    if rec.get("open_questions"):
        lines.append("Open: " + "; ".join(str(x) for x in rec["open_questions"][:6]))
    ws = (rec.get("workspace") or "").strip()
    if ws:
        lines.append("Workspace:")
        lines.append(ws[:80000])
    return "\n".join(lines)


def extra_blocks_for(bot: Any, query: str = "") -> list[tuple[str, dict[str, Any]]]:
    blocks: list[tuple[str, dict[str, Any]]] = []
    rec = load_task_checkpoint(bot)
    text = render_checkpoint(rec or {})
    if text:
        blocks.append(
            (
                text,
                {
                    "id": "task_checkpoint",
                    "section": "active_task",
                    "type": "task",
                    "title": rec.get("goal") if rec else "Active task",
                    "source": "active_task",
                    "reason_codes": ["ACTIVE_TASK", "TASK_CHECKPOINT", "UNRESOLVED_TASK"],
                    "pinned": True,
                },
            )
        )
    try:
        import cognitive_profile as cprof
        import embodied_memory as emem
    except Exception:
        return blocks
    if not cprof.is_embodied(bot):
        return blocks
    facade = emem.for_bot(bot)
    q = query or ""
    if cprof.has_cap(bot, "safety_state"):
        safety = facade.render_safety()
        if safety:
            blocks.append(
                (
                    safety,
                    {
                        "id": "safety_state",
                        "section": "safety_state",
                        "type": "safety",
                        "title": "Safety state",
                        "source": "MiniOS",
                        "reason_codes": ["SAFETY_RELEVANT", "SYSTEM_REQUIRED"],
                        "pinned": True,
                    },
                )
            )
    if cprof.has_cap(bot, "body_state"):
        body = facade.body_block()
        if body:
            blocks.append(
                (
                    body,
                    {
                        "id": "body_state_live",
                        "section": "body_state",
                        "type": "body_state",
                        "title": "Authoritative body state",
                        "source": "MiniOS",
                        "reason_codes": ["CURRENT_BODY_STATE", "SYSTEM_REQUIRED"],
                        "pinned": True,
                    },
                )
            )
    if cprof.has_cap(bot, "world_state"):
        world = facade.render_world()
        if world:
            blocks.append(
                (
                    world,
                    {
                        "id": "world_state_live",
                        "section": "world_state",
                        "type": "world_state",
                        "title": "Current world state",
                        "source": "world_state",
                        "reason_codes": ["CURRENT_WORLD_STATE"],
                        "pinned": True,
                    },
                )
            )
    if cprof.has_cap(bot, "spatial_memory"):
        spatial = facade.render_spatial()
        if spatial:
            blocks.append(
                (
                    spatial,
                    {
                        "id": "spatial_memory",
                        "section": "world_state",
                        "type": "spatial",
                        "title": "Spatial relations",
                        "source": "world_state",
                        "reason_codes": ["CURRENT_WORLD_STATE"],
                        "pinned": False,
                    },
                )
            )
    if cprof.has_cap(bot, "continuous_perception"):
        perc = facade.render_perception()
        if perc:
            blocks.append(
                (
                    perc,
                    {
                        "id": "perception_events",
                        "section": "perception",
                        "type": "perception",
                        "title": "Perception events",
                        "source": "perception",
                        "reason_codes": ["PERCEPTION_RELEVANT"],
                        "pinned": False,
                    },
                )
            )
    if cprof.has_cap(bot, "motor_learning"):
        skill = facade.find_motor_skill(q)
        if skill:
            blocks.append(
                (
                    facade.render_motor(skill),
                    {
                        "id": f"motor_{skill.get('skill')}",
                        "section": "skills",
                        "type": "procedural",
                        "title": skill.get("skill") or "Motor skill",
                        "source": "skill",
                        "reason_codes": ["SKILL_MATCH"]
                        + (["USER_CORRECTION"] if skill.get("corrections") else []),
                        "pinned": False,
                    },
                )
            )
    if cprof.has_cap(bot, "sensorimotor_memory"):
        for ep in facade.retrieve_episodes(q, limit=2):
            blocks.append(
                (
                    facade.render_episode(ep),
                    {
                        "id": str(ep.get("episode_id") or "sensorimotor"),
                        "section": "sensorimotor",
                        "type": "sensorimotor_episode",
                        "title": ep.get("goal") or "Sensorimotor episode",
                        "source": "episodic_memory",
                        "reason_codes": ["PERCEPTION_RELEVANT", "SKILL_MATCH"],
                        "pinned": False,
                    },
                )
            )
    return blocks


def memory_budget(bot: Any) -> botmem.TokenBudget:
    model = str(getattr(bot, "model", "") or "")
    base = botmem.TokenBudget.for_model(model)
    cap = wm.occupancy_capacity(bot) or base.total_context
    pressure = occupancy_pressure(bot)
    available = base.available_for_memory
    if _large_task_pending(bot) and pressure in {wm.PRESSURE_GREEN, wm.PRESSURE_YELLOW, wm.PRESSURE_ORANGE}:
        headroom = int(cap * HEADROOM_FRACTION) if cap else 80000
        available = max(available, min(LARGE_TASK_MEMORY, max(0, (cap or LARGE_TASK_MEMORY) - headroom)))
    if pressure == wm.PRESSURE_RED:
        available = min(available, 8000)
    if pressure == wm.PRESSURE_CRITICAL:
        available = min(available, 4000)
    return botmem.TokenBudget(
        total_context=base.total_context,
        completion_reserve=base.completion_reserve,
        available_for_memory=int(available),
    )


def apply_pressure_policy(bot: Any) -> bool:
    """Compact working memory according to occupancy pressure. Real CONTEXT_COMPACTED only."""
    mgr = getattr(bot, "memory", None)
    if mgr is None:
        return False
    pressure = occupancy_pressure(bot)
    compacted = False
    prev = mgr.compact_threshold_tokens
    mgr.compact_threshold_tokens = PRESSURE_COMPACT_THRESHOLD.get(pressure, prev)
    try:
        before_events = len(mgr.context_events)
        mgr._compact_if_needed()
        compacted = any(
            e.get("type") == "CONTEXT_COMPACTED" for e in list(mgr.context_events)[before_events:]
        )
        if pressure == wm.PRESSURE_CRITICAL:
            if mgr.force_compact():
                compacted = True
            if compacted:
                mgr.context_events.append(
                    {
                        "type": "CONTEXT_REBUILT",
                        "ts": botmem._iso_now(),
                        "reason": "CRITICAL pressure — checkpoint then reconstruct",
                    }
                )
        elif pressure in {wm.PRESSURE_ORANGE, wm.PRESSURE_RED} and not compacted:
            if mgr.force_compact():
                compacted = True
    finally:
        mgr.compact_threshold_tokens = prev
    return compacted


def before_assemble(bot: Any, user_text: str = "") -> None:
    ctx = _load_momentum(bot)
    _sync_checkpoint_from_momentum(bot, user_text, ctx=ctx)
    maybe_complete_task(bot, user_text, ctx=ctx)
    rec = load_task_checkpoint(bot)
    if rec and str(rec.get("status") or "").lower() not in {"done", "completed"}:
        if _user_continues_task(user_text):
            save_task_checkpoint(
                bot,
                {
                    "next_action": rec.get("next_action") or rec.get("current_step"),
                    "status": "RUNNING",
                },
            )
    apply_pressure_policy(bot)


def assemble_for_bot(
    bot: Any,
    messages: list[Any],
    query: str,
    budget: botmem.TokenBudget | None = None,
) -> botmem.AssembledContext:
    mgr = getattr(bot, "memory", None)
    if mgr is None:
        return botmem.AssembledContext(text="", token_count=0)
    rec = load_task_checkpoint(bot)
    goal = (rec or {}).get("goal") if rec else None
    pressure = occupancy_pressure(bot)
    tail_n = CONVERSATION_LIMITS.get(pressure, 8)
    extra = extra_blocks_for(bot, query)
    return mgr.observe_and_assemble(
        messages,
        query,
        budget or memory_budget(bot),
        max_retrieve=3 if not _large_task_pending(bot) else 5,
        min_relevance=0.18 if _large_task_pending(bot) else botmem.DEFAULT_MIN_RELEVANCE,
        max_tail_turns=tail_n,
        task_goal=goal or query,
        extra_blocks=extra,
        exclude_body_observations=True,
    )


def inject_checkpoint_message(payload: dict[str, Any], bot: Any) -> dict[str, Any]:
    """Ensure the active-task checkpoint is in the system turn even when memory inject is skipped."""
    rec = load_task_checkpoint(bot)
    text = render_checkpoint(rec or {})
    if not text:
        return payload
    messages = list(payload.get("messages") or [])
    if messages and isinstance(messages[0], dict) and str(messages[0].get("role")) == "system":
        prev = str(messages[0].get("content") or "")
        if "# Active task checkpoint" not in prev:
            messages[0] = {**messages[0], "content": (prev.rstrip() + "\n\n" + text).strip()}
            payload["messages"] = messages
    return payload
