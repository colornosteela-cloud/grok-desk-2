#!/usr/bin/env python3
"""Virtual-only Teela body service: Qwen tools ↔ HTML simulator.

Does not dispatch Jetson/WBC/PCA9685. The 3D iframe is the body.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import struct
import threading
import time
import uuid
from typing import Any, Callable

SKILLS = (
    "orient_head",
    "raise_arm",
    "lower_arm",
    "wave",
    "neutral_pose",
    "stop",
    "gesture",
)
# Head/arm settles on the HTML twin in ~800ms. Wait so speech uses live joints.
SETTLE_SKILLS = frozenset({"orient_head", "stop", "neutral_pose", "raise_arm", "lower_arm"})
SETTLE_TIMEOUT = 1.2
SETTLE_HTTP_TIMEOUT = 1.5
GESTURES = (
    "greeting",
    "agree",
    "disagree",
    "confused",
    "thinking",
    "excited",
    "point",
    "shrug",
    "listen",
)
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_lock = threading.Lock()
_pending: dict[str, dict[str, Any]] = {}
_last_state: dict[str, dict[str, Any]] = {}
_ws_clients: dict[str, list[Any]] = {}
_emit: Callable[..., None] | None = None


def bind_emit(fn: Callable[..., None]) -> None:
    global _emit
    _emit = fn


def ws_accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key.strip() + _WS_GUID).encode("utf-8")).digest()).decode("ascii")


def encode_ws_text(text: str) -> bytes:
    raw = text.encode("utf-8")
    n = len(raw)
    if n < 126:
        hdr = bytes([0x81, n])
    elif n < 65536:
        hdr = bytes([0x81, 126]) + struct.pack("!H", n)
    else:
        hdr = bytes([0x81, 127]) + struct.pack("!Q", n)
    return hdr + raw


def read_ws_text(rfile: Any) -> str | None:
    hdr = rfile.read(2)
    if not hdr or len(hdr) < 2:
        return None
    opcode = hdr[0] & 0x0F
    masked = bool(hdr[1] & 0x80)
    n = hdr[1] & 0x7F
    if n == 126:
        ext = rfile.read(2)
        if len(ext) < 2:
            return None
        n = struct.unpack("!H", ext)[0]
    elif n == 127:
        ext = rfile.read(8)
        if len(ext) < 8:
            return None
        n = struct.unpack("!Q", ext)[0]
    mask = rfile.read(4) if masked else b""
    data = rfile.read(n) if n else b""
    if masked and mask:
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    if opcode == 8:
        return None
    if opcode == 9:
        return ""
    if opcode != 1:
        return ""
    return data.decode("utf-8", "replace")


def _broadcast(bot_id: str, obj: dict[str, Any]) -> int:
    raw = encode_ws_text(json.dumps(obj, separators=(",", ":")))
    dead: list[Any] = []
    n = 0
    with _lock:
        clients = list(_ws_clients.get(bot_id) or [])
    for sock in clients:
        try:
            sock.sendall(raw)
            n += 1
        except Exception:
            dead.append(sock)
    if dead:
        with _lock:
            cur = _ws_clients.get(bot_id) or []
            _ws_clients[bot_id] = [s for s in cur if s not in dead]
    return n


def register_ws(bot_id: str, sock: Any) -> None:
    with _lock:
        _ws_clients.setdefault(bot_id, []).append(sock)


def unregister_ws(bot_id: str, sock: Any) -> None:
    with _lock:
        cur = _ws_clients.get(bot_id) or []
        _ws_clients[bot_id] = [s for s in cur if s is not sock]


def note_state(bot_id: str, body_state: dict[str, Any] | None) -> None:
    if not isinstance(body_state, dict):
        return
    incoming = dict(body_state)
    with _lock:
        cur = _last_state.get(bot_id) or {}
        incoming_stamp = float(incoming.get("stamp_t") or 0)
        cur_stamp = float(cur.get("stamp_t") or 0)
        if incoming_stamp >= cur_stamp and incoming_stamp > 0:
            _last_state[bot_id] = incoming
            return
        if cur_stamp > 0 and (cur.get("joints") or cur.get("live")):
            return
        _last_state[bot_id] = incoming


def latest_state(bot_id: str) -> dict[str, Any]:
    with _lock:
        return dict(_last_state.get(bot_id) or {"mode": "virtual", "locked": False, "joints": {}, "motion": "idle"})


def joints_of(state: dict[str, Any] | None) -> dict[str, Any]:
    """neck_pan / neck_tilt from HTML virtual state, including head_pan aliases."""
    st = state if isinstance(state, dict) else {}
    joints: dict[str, Any] = {}
    live = st.get("live") if isinstance(st.get("live"), dict) else {}
    raw = st.get("joints") if isinstance(st.get("joints"), dict) else {}
    joints.update(live)
    joints.update(raw)
    if st.get("head_pan") is not None:
        joints["neck_pan"] = st.get("head_pan")
    if st.get("head_tilt") is not None:
        joints["neck_tilt"] = st.get("head_tilt")
    return joints


def apply_to_robot(robot_state: dict[str, Any] | None, virt: dict[str, Any] | None) -> dict[str, Any]:
    """Copy HTML virtual joints onto MiniOS robot_state so I-feel matches the mesh."""
    st = robot_state if isinstance(robot_state, dict) else {}
    virt = virt if isinstance(virt, dict) else {}
    joints = joints_of(virt)
    if not joints:
        return st
    live = dict(st.get("live") or st.get("joints") or {})
    live.update({k: v for k, v in joints.items() if v is not None})
    st["live"] = live
    st["joints"] = dict(live)
    pose = str(virt.get("pose") or "").strip()
    motion = str(virt.get("motion") or "").strip()
    if pose:
        st["pose"] = pose
    if motion:
        st["motion"] = motion
    elif pose != "wave":
        if str(st.get("motion") or "") == "waving":
            st["motion"] = "idle"
    if "waving" in virt:
        st["waving"] = bool(virt.get("waving"))
    elif motion == "waving":
        st["waving"] = True
    elif motion:
        st["waving"] = False
    return st


def head_speech(joints: dict[str, Any] | None) -> str:
    """First-person head line from live neck_pan / neck_tilt. Visual: negative pan = left."""
    joints = joints if isinstance(joints, dict) else {}
    try:
        pan = float(joints.get("neck_pan") or 0)
    except (TypeError, ValueError):
        pan = 0.0
    try:
        tilt = float(joints.get("neck_tilt") or 0)
    except (TypeError, ValueError):
        tilt = 0.0
    if pan <= -12:
        look = "I'm looking left"
    elif pan >= 12:
        look = "I'm looking right"
    else:
        look = "I'm looking straight ahead"
    if tilt >= 8:
        look += ", a little up"
    elif tilt <= -8:
        look += ", a little down"
    return look + "."


def overlay(bot_id: str, state: dict[str, Any] | None) -> dict[str, Any]:
    """Prefer the HTML virtual pose when answering 'where is my hand'."""
    st = dict(state or {})
    virt = latest_state(bot_id) if bot_id else {}
    joints = joints_of(virt)
    if not joints:
        return st
    live = dict(st.get("live") or st.get("joints") or {})
    live.update({k: v for k, v in joints.items() if v is not None})
    st["live"] = live
    st["joints"] = live
    for key in ("pose", "motion", "walk_direction", "heading"):
        if virt.get(key) not in (None, ""):
            st[key] = virt[key]
    if "waving" in virt:
        st["waving"] = bool(virt.get("waving"))
    return st


def stamp_intended(bot_id: str, skill: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write the intended simulated pose so I-feel matches the HTML twin without waiting on WS."""
    import robot_sim as _rs

    skill = str(skill or "").strip().lower()
    params = dict(params or {})
    if skill in GESTURES:
        params.setdefault("gesture", skill)
        skill = "gesture"
    prior = latest_state(bot_id)
    joints = dict(prior.get("joints") or prior.get("live") or {})
    if not joints:
        joints = dict(_rs.POSES.get("home") or {})
    out = dict(prior)
    out["mode"] = "virtual"
    g = str(params.get("gesture") or "").strip().lower()
    if skill == "wave" or (skill == "gesture" and g in {"greeting", "wave", ""}):
        side = "left" if str(params.get("side") or "right").lower() == "left" else "right"
        joints = dict(_rs.POSES.get("home") or {})
        hold = dict(_rs.POSES.get("wave") or {})
        if side == "left":
            hold = {
                "left_shoulder": hold.get("right_shoulder", 24),
                "left_shoulder_out": hold.get("right_shoulder_out", 0),
                "left_elbow": hold.get("right_elbow", 118),
                "left_wrist": hold.get("right_wrist", 0),
                "right_shoulder": 0,
                "right_shoulder_out": 0,
                "right_elbow": 5,
                "right_wrist": 0,
                "neck_pan": 0,
                "neck_tilt": 0,
            }
        else:
            hold = {k: v for k, v in hold.items() if v is not None}
            hold["neck_pan"] = 0
            hold["neck_tilt"] = 0
        joints.update({k: v for k, v in hold.items() if v is not None})
        out.update({"pose": "wave", "motion": "waving", "waving": True, "joints": joints, "live": dict(joints)})
    elif skill in {"stop", "neutral_pose"}:
        home = dict(_rs.POSES.get("home") or {})
        out.update({"pose": "home", "motion": "idle", "waving": False, "joints": home, "live": dict(home)})
    elif skill == "orient_head":
        if params.get("pan_deg") is not None:
            joints["neck_pan"] = float(params.get("pan_deg") or 0)
        if params.get("tilt_deg") is not None:
            joints["neck_tilt"] = float(params.get("tilt_deg") or 0)
        out.update({
            "joints": joints,
            "live": dict(joints),
            "waving": False,
            "motion": "idle",
        })
        if str(out.get("pose") or "") in {"wave", "waving"}:
            out["pose"] = "custom"
    elif skill == "raise_arm":
        side = "left" if str(params.get("side") or "right").lower() == "left" else "right"
        cur = float(joints.get(f"{side}_shoulder") or 0)
        target = 142.0 if cur < 100 else min(165.0, cur + 12.0)
        if params.get("degrees") is not None:
            try:
                target = float(params.get("degrees") or target)
            except (TypeError, ValueError):
                pass
        joints[f"{side}_shoulder"] = target
        joints[f"{side}_elbow"] = 18
        joints[f"{side}_shoulder_out"] = 0
        other = "left" if side == "right" else "right"
        joints[f"{other}_shoulder"] = 0
        joints[f"{other}_shoulder_out"] = 0
        joints[f"{other}_elbow"] = 5
        out.update({"pose": "custom", "motion": "idle", "waving": False, "joints": joints, "live": dict(joints)})
    elif skill == "lower_arm":
        side = "left" if str(params.get("side") or "right").lower() == "left" else "right"
        joints[f"{side}_shoulder"] = 0
        joints[f"{side}_shoulder_out"] = 0
        joints[f"{side}_elbow"] = 5
        out.update({"pose": "custom", "motion": "idle", "waving": False, "joints": joints, "live": dict(joints)})
    elif skill == "gesture" and g == "point":
        side = "left" if str(params.get("side") or "right").lower() == "left" else "right"
        joints[f"{side}_shoulder"] = 78
        joints[f"{side}_elbow"] = 8
        out.update({"pose": "custom", "motion": "idle", "waving": False, "joints": joints, "live": dict(joints)})
    else:
        return out
    out["stamp_t"] = time.time()
    note_state(bot_id, out)
    return out


def seed_boot_pose(bot_id: str) -> dict[str, Any]:
    """Process restart does not resume a mid-gesture. Skills persist elsewhere."""
    return stamp_intended(bot_id, "stop", {})


def note_from_robot(bot_id: str, robot_state: dict[str, Any] | None) -> dict[str, Any]:
    """Keep the virtual overlay in lockstep after MiniOS robot_sim applies a motion."""
    st = robot_state if isinstance(robot_state, dict) else {}
    joints = dict(st.get("live") or st.get("joints") or {})
    motion = str(st.get("motion") or "idle")
    pose = str(st.get("pose") or "")
    waving = bool(st.get("waving")) or motion == "waving" or pose == "wave"
    if motion in {"walking", "idle"} and pose != "wave":
        waving = False
    body = {
        "mode": "virtual",
        "joints": joints,
        "live": dict(joints),
        "pose": pose,
        "motion": motion,
        "waving": waving,
        "walk_direction": st.get("walk_direction"),
        "heading": st.get("heading"),
        "stamp_t": time.time(),
    }
    cur = latest_state(bot_id)
    cur_stamp = float(cur.get("stamp_t") or 0)
    walking = motion in {"walking", "walk"} or pose in {"walk-cycle", "walk"}
    if cur_stamp > 0 and (cur.get("joints") or cur.get("live")) and not walking:
        return cur
    note_state(bot_id, body)
    return body


def complete_action(action_id: str, result: dict[str, Any]) -> None:
    bot_id = ""
    st = result.get("body_state") if isinstance(result, dict) else None
    with _lock:
        slot = _pending.get(action_id)
        if slot:
            slot["result"] = result
            slot["event"].set()
            bot_id = str(slot.get("bot_id") or "")
    if isinstance(st, dict) and bot_id:
        note_state(bot_id, st)


def settle_timeout(skill: str, explicit: float | None = None) -> float:
    """Seconds to wait for live joints. Wave/gesture stay fire-and-forget."""
    if explicit is not None and float(explicit) > 0:
        return float(explicit)
    if str(skill or "").strip().lower() in SETTLE_SKILLS:
        return SETTLE_HTTP_TIMEOUT
    return 0.0


def wait_settle(
    bot_id: str,
    timeout: float = SETTLE_TIMEOUT,
    want: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Block until the HTML twin reports action_result, or timeout.

    Idle (no pending action) returns immediately with latest_state.
    """
    timeout = max(0.0, float(timeout or 0))
    deadline = time.time() + timeout
    events: list[threading.Event] = []
    with _lock:
        for slot in _pending.values():
            if str(slot.get("bot_id") or "") != str(bot_id or ""):
                continue
            ev = slot.get("event")
            if isinstance(ev, threading.Event):
                events.append(ev)
    for ev in events:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        ev.wait(remaining)
    if want:
        while time.time() < deadline:
            if not move_still_needed(want, latest_state(bot_id)):
                break
            time.sleep(0.04)
    return latest_state(bot_id)


def submit_action(
    bot_id: str,
    skill: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 0.0,
) -> dict[str, Any]:
    """Dispatch a virtual move. timeout=0 returns immediately (wave/speech overlap).

    Callers that need live joints (look/orient) pass timeout so we wait for
    action_result before returning.
    """
    skill = str(skill or "").strip().lower()
    params = dict(params or {})
    if skill in GESTURES:
        params.setdefault("gesture", skill)
        skill = "gesture"
    if skill == "gesture":
        g = str(params.get("gesture") or "").strip().lower()
        if g not in GESTURES:
            return {"status": "rejected", "reason": f"unknown_gesture:{g}", "mode": "virtual"}
        params["gesture"] = g
    elif skill not in SKILLS and skill not in {"stop", "neutral_pose"}:
        return {"status": "rejected", "reason": f"unknown_skill:{skill}", "mode": "virtual"}
    st = stamp_intended(bot_id, skill, params)
    if skill == "raise_arm":
        side = "left" if str(params.get("side") or "right").lower() == "left" else "right"
        deg = (st.get("joints") or st.get("live") or {}).get(f"{side}_shoulder")
        if deg is not None:
            params["degrees"] = float(deg)
    action_id = "act_" + uuid.uuid4().hex[:10]
    ev = threading.Event()
    with _lock:
        _pending[action_id] = {"event": ev, "result": None, "bot_id": bot_id, "t0": time.time()}
    msg = {
        "type": "body_action",
        "action_id": action_id,
        "skill": skill,
        "parameters": params,
        "mode": "virtual",
    }
    sent = _broadcast(bot_id, msg)
    if _emit:
        try:
            _emit({"type": "virtual.body_action", "bot_id": bot_id, **msg})
        except Exception:
            pass
    st["action"] = {
        "id": action_id,
        "skill": str(params.get("gesture") or skill),
        "status": "executing",
    }
    note_state(bot_id, st)
    wait = max(0.0, float(timeout or 0))
    if wait > 0 and ev.wait(wait):
        with _lock:
            slot = _pending.pop(action_id, None) or {}
        result = slot.get("result") if isinstance(slot.get("result"), dict) else {}
        result.setdefault("action_id", action_id)
        result.setdefault("skill", skill)
        result.setdefault("mode", "virtual")
        return result
    return {
        "status": "executing",
        "action_id": action_id,
        "skill": skill,
        "parameters": params,
        "mode": "virtual",
        "ws_clients": sent,
        "started": True,
        "body_state": st,
    }


def handle_client_message(bot_id: str, obj: dict[str, Any]) -> None:
    kind = str(obj.get("type") or "")
    if kind == "body_state":
        note_state(bot_id, obj)
        return
    if kind == "action_result":
        aid = str(obj.get("action_id") or "")
        if aid:
            complete_action(aid, obj)
        st = obj.get("body_state")
        if isinstance(st, dict):
            note_state(bot_id, st)


def _virtual_side(text: str, default: str = "right") -> str:
    """User's words map onto Teela's own left/right, not screen-mirror."""
    has_l = bool(re.search(r"\bleft\b", text or ""))
    has_r = bool(re.search(r"\bright\b", text or ""))
    if has_l and not has_r:
        return "left"
    if has_r and not has_l:
        return "right"
    return default


def _little(text: str) -> bool:
    return bool(re.search(r"\b(?:a little|little|bit|slightly|slight|softly)\b", text or ""))


def teela_args_from_intent(intent: str) -> tuple[str, dict[str, Any]] | None:
    """Natural language → virtual body tool. Prefer this over MiniOS mirroring."""
    t = " ".join((intent or "").lower().split())
    if not t:
        return None
    if re.search(r"\b(?:stop|halt|freeze|hold still)\b", t) and not re.search(r"\b(?:don't|do not|never)\s+stop\b", t):
        if re.search(r"\b(?:and|then)\b.{0,24}\b(?:walk|raise|bow|sit)\b", t):
            return None
        return "bot_desktop__teela_stop", {"skill": "stop"}
    if re.search(
        r"\b(?:wave\s+(?:hello|hi|hey)|wave\s+at\s+me|hello\s+wave|greet(?:ing)?(?:\s+me)?)\b",
        t,
    ):
        return "bot_desktop__teela_gesture", {"gesture": "greeting", "side": _virtual_side(t, "right")}
    if re.search(r"\b(?:nod|agree|that's right|looks good)\b", t) and not re.search(r"\b(?:arm|hand|walk)\b", t):
        return "bot_desktop__teela_gesture", {"gesture": "agree"}
    if re.search(r"\b(?:shake your head|disagree|nope)\b", t):
        return "bot_desktop__teela_gesture", {"gesture": "disagree"}
    if re.search(r"\b(?:confused|don't understand|do not understand|puzzled)\b", t):
        return "bot_desktop__teela_gesture", {"gesture": "confused"}
    if re.search(r"\b(?:think(?:ing)?(?:\s+about it)?|let me think|hmm)\b", t) and re.search(
        r"\b(?:think|gesture|show|look)\b", t
    ):
        return "bot_desktop__teela_gesture", {"gesture": "thinking"}
    if re.search(r"\b(?:excited|cheer|yay)\b", t):
        return "bot_desktop__teela_gesture", {"gesture": "excited"}
    if re.search(r"\b(?:shrug|don't know|do not know|dunno)\b", t):
        return "bot_desktop__teela_gesture", {"gesture": "shrug"}
    if re.search(r"\b(?:listen(?:ing)?|i(?:'m| am) listening)\b", t) and re.search(
        r"\b(?:listen|gesture|look|face)\b", t
    ):
        return "bot_desktop__teela_gesture", {"gesture": "listen"}
    if re.search(r"\b(?:point|over there)\b", t) and not re.search(r"\bwalk\b", t):
        return "bot_desktop__teela_gesture", {"gesture": "point", "side": _virtual_side(t, "right")}
    if re.search(r"\bwave\b", t):
        return "bot_desktop__teela_body_action", {"skill": "wave", "side": _virtual_side(t, "right")}
    if re.search(r"\b(?:neutral|stand (?:up|straight)|reset(?: your)?(?: pose| body)?)\b", t):
        return "bot_desktop__teela_body_action", {"skill": "neutral_pose"}
    # Walk beats "turn … left" (that regex is for the head). "Turn and walk left" is a walk.
    if re.search(r"\bwalk(?:ing)?\b", t) and not re.search(
        r"\b(?:stop|halt)\b.{0,24}\b(?:and|then)\b.{0,24}\bwalk", t
    ):
        direction = "place"
        if re.search(r"\b(?:back(?:wards?)?|away(?:\s+from me)?|north)\b", t):
            direction = "back"
        elif re.search(r"\b(?:toward(?:s)? me|forward|forwards|south)\b", t):
            direction = "south"
        elif re.search(r"\beast\b|\bleft\b", t):
            direction = "left"
        elif re.search(r"\bwest\b|\bright\b", t):
            direction = "right"
        return "bot_desktop__robot_motion", {"cmd": "walk", "direction": direction}
    if re.search(r"right shoulder", t) and re.search(r"\b(?:head|look|turn|face|toward|towards)\b", t):
        return "bot_desktop__teela_body_action", {"skill": "orient_head", "pan_deg": 25.0}
    if re.search(r"left shoulder", t) and re.search(r"\b(?:head|look|turn|face|toward|towards)\b", t):
        return "bot_desktop__teela_body_action", {"skill": "orient_head", "pan_deg": -25.0}
    if re.search(r"\bwalk\b", t):
        return None
    if re.search(r"\b(?:look|turn|face|pan)\b.{0,28}\bleft\b|\bhead\b.{0,16}\bleft\b", t):
        pan = -12.0 if _little(t) else -25.0
        return "bot_desktop__teela_body_action", {"skill": "orient_head", "pan_deg": pan}
    if re.search(r"\b(?:look|turn|face|pan)\b.{0,28}\bright\b|\bhead\b.{0,16}\bright\b", t):
        pan = 12.0 if _little(t) else 25.0
        return "bot_desktop__teela_body_action", {"skill": "orient_head", "pan_deg": pan}
    if re.search(
        r"\b(?:look|face|turn)\b.{0,16}\b(?:straight|center|ahead|forward|at me)\b|"
        r"\bhead\b.{0,12}\b(?:straight|center|forward)\b|"
        r"\b(?:straighten|center)\b.{0,12}\bhead\b",
        t,
    ):
        return "bot_desktop__teela_body_action", {"skill": "orient_head", "pan_deg": 0.0, "tilt_deg": 0.0}
    if re.search(r"\b(?:look|tilt|face|chin)\b.{0,20}\bup\b|\bhead\b.{0,12}\bup\b", t):
        tilt = 10.0 if _little(t) else 18.0
        return "bot_desktop__teela_body_action", {"skill": "orient_head", "tilt_deg": tilt}
    if re.search(r"\b(?:look|tilt|face|chin)\b.{0,20}\bdown\b|\bhead\b.{0,12}\bdown\b", t):
        tilt = -8.0 if _little(t) else -16.0
        return "bot_desktop__teela_body_action", {"skill": "orient_head", "tilt_deg": tilt}
    if re.search(
        r"\b(?:lower|drop)\b.{0,24}\b(?:arm|hand)\b|"
        r"\bput.{0,24}\b(?:arm|hand).{0,12}\bdown\b|"
        r"\b(?:arm|hand).{0,12}\bdown\b|"
        r"\bput (?:it|your (?:arm|hand)) back\b",
        t,
    ):
        return "bot_desktop__teela_body_action", {"skill": "lower_arm", "side": _virtual_side(t, "right")}
    if re.search(
        r"\b(?:raise|lift|put up)\b.{0,24}\b(?:arm|hand)\b|"
        r"\b(?:arm|hand).{0,16}\b(?:up|overhead)\b",
        t,
    ):
        return "bot_desktop__teela_body_action", {"skill": "raise_arm", "side": _virtual_side(t, "right")}
    return None


def teela_args_from_motor(
    cmd: dict[str, Any] | None, intent: str = ""
) -> tuple[str, dict[str, Any]] | None:
    """Map chat + MiniOS infer_command onto teela_body_action / teela_gesture / teela_stop.

    Live MiniOS: negative neck_pan is look left, positive is look right.
    """
    hit = teela_args_from_intent(intent)
    if hit:
        return hit
    if not isinstance(cmd, dict):
        return None
    pose = str(cmd.get("pose") or "").strip().lower()
    kind = str(cmd.get("cmd") or "").strip().lower()
    joint = str(cmd.get("joint") or "").strip().lower()
    text = " ".join((intent or "").lower().split())
    if pose == "wave" or kind == "wave":
        return "bot_desktop__teela_body_action", {"skill": "wave", "side": _virtual_side(text, "right")}
    if pose in {"home", "neutral"} or kind in {"neutral"}:
        return "bot_desktop__teela_body_action", {"skill": "neutral_pose"}
    if kind in {"stop", "stop_demo"}:
        return "bot_desktop__teela_stop", {"skill": "stop"}
    if kind == "plan":
        out: dict[str, Any] = {"cmd": "plan", "steps": cmd.get("steps") or []}
        if cmd.get("why"):
            out["why"] = cmd.get("why")
        if cmd.get("direction"):
            out["direction"] = cmd.get("direction")
        return "bot_desktop__robot_motion", out
    if kind in {"walk", "start_walk"} or kind.startswith("walk"):
        return "bot_desktop__robot_motion", {
            "cmd": "walk",
            "direction": str(cmd.get("direction") or "place"),
        }
    if joint == "neck_pan":
        val = cmd.get("value")
        direction = str(cmd.get("dir") or cmd.get("direction") or "").lower()
        if val is None:
            pan = -25.0 if direction != "right" else 25.0
        else:
            v = float(val)
            # robot_sim left=+40; live skeleton left is negative pan.
            pan = -v
            if abs(abs(v) - 40) < 0.51:
                pan = -25.0 if v > 0 else 25.0
        return "bot_desktop__teela_body_action", {"skill": "orient_head", "pan_deg": pan}
    if joint == "neck_tilt":
        val = cmd.get("value")
        tilt = 18.0 if val is None else float(val)
        return "bot_desktop__teela_body_action", {"skill": "orient_head", "tilt_deg": tilt}
    joints = cmd.get("joints") if isinstance(cmd.get("joints"), dict) else {}
    if float(joints.get("right_shoulder") or 0) >= 120 and float(joints.get("left_shoulder") or 0) < 40:
        return "bot_desktop__teela_body_action", {"skill": "raise_arm", "side": "right"}
    if float(joints.get("left_shoulder") or 0) >= 120 and float(joints.get("right_shoulder") or 0) < 40:
        return "bot_desktop__teela_body_action", {"skill": "raise_arm", "side": "left"}
    if kind in {"joint", "joints"} and (
        "right_shoulder" in joints or "left_shoulder" in joints
    ):
        if float(joints.get("right_shoulder") or 0) < 20 and float(joints.get("left_shoulder") or 0) < 20:
            return "bot_desktop__teela_body_action", {
                "skill": "lower_arm",
                "side": "right" if "right_shoulder" in joints else "left",
            }
    return None


def same_request(done: dict[str, Any] | None, want: dict[str, Any] | None) -> bool:
    """True when a prior teela_* call this turn is already the requested skill."""
    done = done if isinstance(done, dict) else {}
    want = want if isinstance(want, dict) else {}
    if not want:
        return False
    params = done.get("parameters") if isinstance(done.get("parameters"), dict) else {}
    merged = {**params, **{k: v for k, v in done.items() if k != "parameters"}}

    def _move_name(d: dict[str, Any]) -> str:
        g = str(d.get("gesture") or "").strip().lower()
        if g:
            return g
        return str(d.get("skill") or "").strip().lower()

    skill = _move_name(want)
    got = _move_name(merged)
    if not skill or got != skill:
        return False
    if skill == "orient_head":
        for key in ("pan_deg", "tilt_deg"):
            if want.get(key) is None:
                continue
            if merged.get(key) is None:
                return False
            try:
                if abs(float(merged[key]) - float(want[key])) > 6:
                    return False
            except (TypeError, ValueError):
                return False
        return True
    if skill in {"wave", "raise_arm", "lower_arm", "greeting", "point"}:
        # The HTML skills default to the right hand when side is omitted.
        side = str(want.get("side") or "right").strip().lower()
        got_side = str(merged.get("side") or "right").strip().lower()
        if side != got_side:
            return False
    return True


def move_still_needed(args: dict[str, Any] | None, body_state: dict[str, Any] | None) -> bool:
    """True when the virtual pose does not yet match this skill."""
    args = args if isinstance(args, dict) else {}
    st = body_state if isinstance(body_state, dict) else {}
    joints = joints_of(st)
    skill = str(args.get("skill") or args.get("gesture") or "").strip().lower()
    pose = str(st.get("pose") or "").strip().lower()
    motion = str(st.get("motion") or "").strip().lower()

    def j(name: str) -> float:
        try:
            return float(joints.get(name) or 0)
        except (TypeError, ValueError):
            return 0.0

    if skill == "orient_head":
        # Missing joints are not pan=0. Empty state must not skip look-straight.
        if not joints:
            return True
        if args.get("pan_deg") is not None and abs(j("neck_pan") - float(args["pan_deg"])) > 6:
            return True
        if args.get("tilt_deg") is not None and abs(j("neck_tilt") - float(args["tilt_deg"])) > 6:
            return True
        return False
    if skill in {"wave", "greeting"}:
        if st.get("waving") is True or motion == "waving":
            right = j("right_elbow") >= 70 and 12 <= j("right_shoulder") < 85
            left = j("left_elbow") >= 70 and 12 <= j("left_shoulder") < 85
            side = str(args.get("side") or "right").strip().lower()
            return not (left if side == "left" else right)
        return True
    if skill == "raise_arm":
        side = str(args.get("side") or "right")
        return j(f"{side}_shoulder") < 100
    if skill == "lower_arm":
        side = str(args.get("side") or "right")
        return j(f"{side}_shoulder") > 25
    if skill in {"stop", "neutral_pose"}:
        return motion not in {"idle", "stopped", ""}
    return True
