#!/usr/bin/env python3
"""Track concurrent body / agent / speech work for one Teela brain.

Virtual body only. Does not dispatch physical motors.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from typing import Any

import virtual_body

_lock = threading.Lock()
_tasks: dict[str, list[dict[str, Any]]] = {}
_MAX = 16

_AGENT_RE = re.compile(
    r"\b(?:inspect(?:ing)?|check(?:ing)?|status|diagnostic|server|logs?|system information|"
    r"uname|disk|cpu|shell|files?|browser|search|code|tests?|run the|"
    r"pull up|look(?:ing)? at (?:the )?(?:logs?|server|system)|"
    r"what(?:'s| is) (?:on|wrong with))\b",
    re.I,
)
_SYSTEM_RE = re.compile(
    r"\b(?:"
    r"(?:full\s+)?system\s+(?:check|scan|work|status|health|diagnostics?|information)|"
    r"hardware\s+(?:check|scan|status|health)|"
    r"self[-\s]?test|"
    r"diagnostics?|"
    r"(?:run|perform|do)\s+(?:a\s+|an\s+)?(?:full\s+)?(?:system\s+|hardware\s+)?(?:scan|check)|"
    r"check(?:ing)?\s+(?:yourself|the\s+system|the\s+main\s+system|teela-brain|your\s+(?:system|systems|body|hardware|mesh|health|motors?|twin|workspace|desk|minios))|"
    r"scan\s+(?:the\s+)?(?:system|hardware|body|mesh|workspace|host)|"
    r"(?:workspace|minios|mini-?os|main\s+system|teela-brain|host)\s+(?:check|scan|status|health)|"
    r"inspect(?:ing)?\s+(?:your\s+)?(?:system|systems|hardware|mesh|body)(?:\s+information)?|"
    r"(?:mesh|jetson|wbc|e-?stop|observer)\s+(?:check|status|health)|"
    r"is\s+(?:the\s+)?(?:mesh|jetson|hardware|body|wbc)\s+(?:ok|up|connected|attached)|"
    r"are\s+you\s+(?:healthy|operational)|"
    r"system work"
    r")\b",
    re.I,
)
_WHILE_RE = re.compile(
    r"\bwhile\b.{0,48}\b(?:look|wave|turn|raise|point|gesture|face)\b|"
    r"\b(?:look|wave|turn|raise|point|face)\b.{0,48}\bwhile\b",
    re.I,
)
_AFTER_RE = re.compile(
    r"\b(?:when (?:you(?:'re| are)? )?(?:done|finished)|"
    r"after (?:you )?(?:check|inspect|finish)|"
    r"and then (?:wave|look|turn)|"
    r"wave when)\b",
    re.I,
)


def looks_like_system_work(text: str) -> bool:
    """True when they want Teela's own health/mesh/diagnostics, not a chat or a move."""
    t = " ".join((text or "").lower().split())
    return bool(t and _SYSTEM_RE.search(t))


def classify(text: str) -> dict[str, Any]:
    """Is this body-only, agent-only, or one plan with both?"""
    t = " ".join((text or "").lower().split())
    body = bool(virtual_body.teela_args_from_intent(t))
    if not body:
        body = bool(
            re.search(
                r"\b(?:look|wave|turn your head|raise|gesture|point|nod|shrug)\b",
                t,
            )
        )
    system = looks_like_system_work(t)
    agent = bool(_AGENT_RE.search(t)) or system
    if body and agent:
        mode = "after" if _AFTER_RE.search(t) else "parallel"
    elif body:
        mode = "body"
    elif agent:
        mode = "agent"
    else:
        mode = "none"
    return {"mode": mode, "body": body, "agent": agent, "system": system}


def note(
    bot_id: str,
    kind: str,
    action: str,
    status: str = "running",
    **extra: Any,
) -> dict[str, Any]:
    task = {
        "id": extra.pop("id", None) or ("t_" + uuid.uuid4().hex[:8]),
        "kind": str(kind or "agent"),
        "action": str(action or ""),
        "status": str(status or "running"),
        "t": time.time(),
        **extra,
    }
    with _lock:
        cur = _tasks.setdefault(bot_id, [])
        cur.insert(0, task)
        del cur[_MAX:]
    return task


def complete(bot_id: str, task_id: str, status: str = "completed") -> None:
    with _lock:
        for task in _tasks.get(bot_id) or []:
            if task.get("status") not in {"running", "executing", "speaking"}:
                continue
            if task_id and task.get("id") != task_id:
                continue
            if not task_id and task.get("kind") != "agent":
                continue
            task["status"] = status
            task["done"] = time.time()
            if task_id:
                return


def snapshot(bot_id: str) -> dict[str, Any]:
    body = virtual_body.latest_state(bot_id)
    with _lock:
        tasks = [dict(t) for t in (_tasks.get(bot_id) or [])]
    active = [
        t
        for t in tasks
        if str(t.get("status") or "") in {"running", "executing", "speaking"}
    ]
    you: list[str] = []
    for t in active:
        kind = t.get("kind")
        act = t.get("action") or ""
        if kind == "body":
            you.append(f"moving: {act}")
        elif kind == "agent":
            you.append(f"working: {act}")
        elif kind == "speech":
            you.append("speaking")
    joints = body.get("joints") if isinstance(body.get("joints"), dict) else {}
    pan = joints.get("neck_pan")
    if pan is None:
        pan = body.get("head_pan")
    try:
        pan_f = float(pan or 0)
    except (TypeError, ValueError):
        pan_f = 0.0
    if pan_f <= -12:
        you.append("facing left")
    elif pan_f >= 12:
        you.append("facing right")
    else:
        you.append("facing ahead")
    return {
        "mode": "virtual",
        "you_are": you or ["idle"],
        "body": {
            "pose": body.get("pose"),
            "motion": body.get("motion"),
            "head_pan": pan_f,
            "action": body.get("action"),
        },
        "tasks": active,
        "recent": tasks[:8],
    }


def spoken_now(bot_id: str, text: str) -> None:
    note(bot_id, "speech", (text or "")[:80], status="speaking")


def inspect_system_text() -> str:
    import subprocess

    chunks: list[str] = []
    for cmd in (
        ["uname", "-a"],
        ["uptime"],
        ["free", "-h"],
    ):
        try:
            out = subprocess.check_output(cmd, text=True, timeout=4, stderr=subprocess.STDOUT)
            chunks.append(out.strip())
        except Exception as e:
            chunks.append(f"{' '.join(cmd)}: {e}")
    return "\n".join(chunks)


def inspect_then_wave(bot_id: str, intent: str) -> str | None:
    """Deterministic after-plan: inspect now, wave when that returns."""
    if classify(intent).get("mode") != "after":
        return None
    t = " ".join((intent or "").lower().split())
    if not re.search(r"\bwave\b", t):
        return None
    if not re.search(r"\b(?:inspect|system information|diagnostic|server status)\b", t):
        return None
    note(bot_id, "agent", "inspect_system", status="running")
    info = inspect_system_text()
    complete(bot_id, "", status="completed")
    started = virtual_body.submit_action(bot_id, "wave", {"skill": "wave", "side": "right"})
    note(
        bot_id,
        "body",
        "wave",
        status=str(started.get("status") or "executing"),
        id=started.get("action_id"),
    )
    spoken_now(bot_id, "inspect then wave")
    return (
        "I checked the system:\n"
        f"{info}\n\n"
        "That's done — waving now."
    )
