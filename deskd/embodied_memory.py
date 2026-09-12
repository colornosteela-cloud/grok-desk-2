"""Embodied memory extensions on the shared MemoryManager core.

World, perception events, sensorimotor episodes, spatial relations, safety,
and motor-skill validation. Live body remains in body_state (MiniOS).
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import body_state as bs
import cognitive_profile as cprof

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_DEFAULT_TTL_S = 45.0
_MAX_EVENTS = 80
_MAX_EPISODES = 120


def _iso(ts: float | None = None) -> str:
    t = time.time() if ts is None else float(ts)
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now() -> float:
    return time.time()


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        cur = _LOCKS.get(key)
        if cur is None:
            cur = threading.Lock()
            _LOCKS[key] = cur
        return cur


def _root_for(bot: Any) -> Path:
    ws = getattr(bot, "workspace", None)
    if ws:
        return Path(str(ws)) / ".memory"
    root = getattr(bot, "root", None)
    return Path(str(root or ".")) / ".memory"


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


class EmbodiedMemory:
    """Per-bot embodied stores. Missing files mean Unavailable, never fake zeros."""

    def __init__(self, bot: Any) -> None:
        self.bot = bot
        self.root = _root_for(bot)
        self._key = str(getattr(bot, "id", "") or self.root)
        self._lock = _lock_for(self._key)

    def _world_path(self) -> Path:
        return self.root / "world_state.json"

    def _events_path(self) -> Path:
        return self.root / "perception_events.json"

    def _episodes_path(self) -> Path:
        return self.root / "sensorimotor_episodes.json"

    def _spatial_path(self) -> Path:
        return self.root / "spatial.json"

    def _motor_path(self) -> Path:
        return self.root / "motor_skills.json"

    def _safety_path(self) -> Path:
        return self.root / "safety_state.json"

    # --- world ---
    def observe(
        self,
        key: str,
        *,
        kind: str = "object",
        attrs: dict[str, Any] | None = None,
        visible: bool = True,
        confidence: float = 0.9,
        source: str = "CAMERA_LEFT",
        provenance: str = "OBSERVATION",
        ttl_s: float = _DEFAULT_TTL_S,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now() if now is None else float(now)
        with self._lock:
            rec = _read_json(self._world_path(), {"revision": 0, "entities": {}})
            entities = rec.setdefault("entities", {})
            rec["revision"] = int(rec.get("revision") or 0) + 1
            rec["timestamp"] = _iso(ts)
            entities[str(key)] = {
                "id": str(key),
                "kind": str(kind),
                "visible": bool(visible),
                "confidence": float(confidence),
                "source": str(source),
                "provenance": str(provenance),
                "observed_at": _iso(ts),
                "last_confirmed": _iso(ts),
                "observed_ts": ts,
                "last_confirmed_ts": ts,
                "ttl_s": float(ttl_s),
                "expires_at": _iso(ts + float(ttl_s)),
                **dict(attrs or {}),
            }
            rec["entities"] = entities
            _write_json(self._world_path(), rec)
            return dict(entities[str(key)])

    def mark_stale(self, key: str, *, reason: str = "expired") -> None:
        with self._lock:
            rec = _read_json(self._world_path(), {"revision": 0, "entities": {}})
            ent = (rec.get("entities") or {}).get(str(key))
            if not isinstance(ent, dict):
                return
            ent["visible"] = False
            ent["stale"] = True
            ent["stale_reason"] = reason
            ent["last_confirmed"] = ent.get("last_confirmed")
            rec["revision"] = int(rec.get("revision") or 0) + 1
            rec["timestamp"] = _iso()
            rec.setdefault("entities", {})[str(key)] = ent
            _write_json(self._world_path(), rec)

    def current_world(self, *, now: float | None = None) -> dict[str, Any] | None:
        path = self._world_path()
        if not path.is_file():
            return None
        ts = _now() if now is None else float(now)
        rec = _read_json(path, None)
        if not isinstance(rec, dict):
            return None
        live: dict[str, Any] = {}
        stale: dict[str, Any] = {}
        for key, ent in (rec.get("entities") or {}).items():
            if not isinstance(ent, dict):
                continue
            last = float(ent.get("last_confirmed_ts") or ent.get("observed_ts") or 0)
            ttl = float(ent.get("ttl_s") or _DEFAULT_TTL_S)
            expired = bool(ent.get("stale")) or (ts - last) > ttl or not ent.get("visible", True)
            if expired:
                stale[str(key)] = {**ent, "currently_visible": False, "fresh": False}
            else:
                live[str(key)] = {**ent, "currently_visible": True, "fresh": True}
        if not live and not stale and not rec.get("entities"):
            return None
        return {
            "revision": rec.get("revision"),
            "timestamp": rec.get("timestamp"),
            "source": "WORLD_STATE",
            "people": {k: v for k, v in live.items() if v.get("kind") == "person"},
            "objects": {k: v for k, v in live.items() if v.get("kind") != "person"},
            "stale": stale,
            "hazards": [v for v in live.values() if v.get("hazard")],
        }

    def render_world(self) -> str:
        world = self.current_world()
        if not world:
            return ""
        lines = ["[TEELA WORLD NOW]", f"Revision: {world.get('revision') or 0}"]
        objs = world.get("objects") or {}
        people = world.get("people") or {}
        if people:
            lines.append("People: " + ", ".join(f"{k} visible" for k in people))
        if objs:
            bits = []
            for k, v in objs.items():
                loc = v.get("location") or v.get("relative_position") or ""
                bits.append(f"{k}{' at ' + str(loc) if loc else ''} conf={v.get('confidence')}")
            lines.append("Objects: " + "; ".join(bits))
        if not people and not objs:
            lines.append("Current scene: empty (no fresh observations)")
        stale = world.get("stale") or {}
        if stale:
            lines.append("Stale (not current): " + ", ".join(stale.keys()))
        return "\n".join(lines)

    # --- perception events ---
    def ingest_frame(self, frame_id: Any, description: str = "") -> None:
        """Raw frames are not events and are not stored."""
        with self._lock:
            rec = _read_json(self._events_path(), {"events": [], "last_frame": None, "dropped_frames": 0})
            rec["dropped_frames"] = int(rec.get("dropped_frames") or 0) + 1
            rec["last_frame"] = {"id": frame_id, "ts": _iso()}
            _write_json(self._events_path(), rec)

    def record_event(
        self,
        *,
        type: str,  # noqa: A002
        subject: str = "",
        action: str = "",
        object: str = "",  # noqa: A002
        confidence: float = 0.9,
        source: str = "CAMERA_LEFT",
        modality: str = "vision",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        sig = (str(type), str(subject), str(action), str(object))
        with self._lock:
            rec = _read_json(self._events_path(), {"events": [], "last_sig": None, "dropped_frames": 0})
            if rec.get("last_sig") == list(sig):
                rec["suppressed"] = int(rec.get("suppressed") or 0) + 1
                _write_json(self._events_path(), rec)
                return None
            event = {
                "event_id": f"pev_{int(_now() * 1000) % 10_000_000}_{len(rec.get('events') or [])}",
                "timestamp": _iso(),
                "type": str(type),
                "subject": str(subject),
                "action": str(action),
                "object": str(object),
                "confidence": float(confidence),
                "source": {"modality": modality, "camera": source, "name": source},
                "provenance": "OBSERVATION",
            }
            if extra:
                event.update(extra)
            events = list(rec.get("events") or [])
            events.append(event)
            rec["events"] = events[-_MAX_EVENTS:]
            rec["last_sig"] = list(sig)
            _write_json(self._events_path(), rec)
            return event

    def events(self, limit: int = 8) -> list[dict[str, Any]]:
        rec = _read_json(self._events_path(), None)
        if not isinstance(rec, dict):
            return []
        rows = [e for e in (rec.get("events") or []) if isinstance(e, dict)]
        return rows[-max(0, int(limit)) :]

    def render_perception(self, limit: int = 6) -> str:
        rows = self.events(limit=limit)
        if not rows:
            return ""
        lines = ["[TEELA PERCEPTION EVENTS]"]
        for e in rows:
            bits = [e.get("timestamp") or "", e.get("type") or ""]
            if e.get("subject"):
                bits.append(str(e.get("subject")))
            if e.get("action"):
                bits.append(str(e.get("action")))
            if e.get("object"):
                bits.append(str(e.get("object")))
            lines.append(" ".join(str(b) for b in bits if b))
        return "\n".join(lines)

    def perception_stats(self) -> dict[str, Any] | None:
        path = self._events_path()
        if not path.is_file():
            return None
        rec = _read_json(path, {})
        events = rec.get("events") or []
        return {
            "event_count": len(events),
            "dropped_frames": int(rec.get("dropped_frames") or 0),
            "suppressed": int(rec.get("suppressed") or 0),
        }

    # --- sensorimotor episodes ---
    def record_episode(self, fields: dict[str, Any]) -> dict[str, Any]:
        ep = {
            "type": "sensorimotor_episode",
            "episode_id": fields.get("episode_id") or f"sme_{int(_now() * 1000) % 10_000_000}",
            "timestamp": _iso(),
            "goal": fields.get("goal") or "",
            "before": fields.get("before") or {},
            "action": fields.get("action") or {},
            "actual_result": fields.get("actual_result") or fields.get("actual") or {},
            "perception_after": fields.get("perception_after") or fields.get("after") or {},
            "outcome": fields.get("outcome") or {},
            "metrics": fields.get("metrics") or {},
            "source": fields.get("source") or "MINIOS",
            "provenance": fields.get("provenance") or "OBSERVATION",
            "user_feedback": fields.get("user_feedback"),
            "software_version": fields.get("software_version") or "teela-embodied-1",
        }
        if not self._should_persist(ep) and not fields.get("force"):
            ep["persisted"] = False
            return ep
        with self._lock:
            rec = _read_json(self._episodes_path(), {"episodes": []})
            rows = list(rec.get("episodes") or [])
            rows.append(ep)
            rec["episodes"] = rows[-_MAX_EPISODES:]
            _write_json(self._episodes_path(), rec)
        ep["persisted"] = True
        return ep

    def _should_persist(self, ep: dict[str, Any]) -> bool:
        outcome = ep.get("outcome") if isinstance(ep.get("outcome"), dict) else {}
        if outcome.get("success") is False:
            return True
        if ep.get("user_feedback") or outcome.get("user_correction"):
            return True
        if outcome.get("safety"):
            return True
        if outcome.get("first_success") or outcome.get("novel"):
            return True
        metrics = ep.get("metrics") if isinstance(ep.get("metrics"), dict) else {}
        if metrics.get("improved"):
            return True
        return True  # explicit record_episode from callers is already a chosen episode

    def retrieve_episodes(self, goal: str, limit: int = 3) -> list[dict[str, Any]]:
        rec = _read_json(self._episodes_path(), None)
        if not isinstance(rec, dict):
            return []
        q = (goal or "").strip().lower()
        rows = [e for e in (rec.get("episodes") or []) if isinstance(e, dict)]
        if q:
            ranked = [e for e in rows if q in str(e.get("goal") or "").lower() or str(e.get("goal") or "").lower() in q]
            if not ranked:
                toks = {t for t in q.replace("-", " ").split() if len(t) > 2}
                ranked = [
                    e
                    for e in rows
                    if toks & set(str(e.get("goal") or "").lower().replace("-", " ").split())
                ]
            rows = ranked or rows
        return list(reversed(rows))[: max(0, int(limit))]

    def render_episode(self, ep: dict[str, Any]) -> str:
        action = ep.get("action") if isinstance(ep.get("action"), dict) else {}
        actual = ep.get("actual_result") if isinstance(ep.get("actual_result"), dict) else {}
        outcome = ep.get("outcome") if isinstance(ep.get("outcome"), dict) else {}
        metrics = ep.get("metrics") if isinstance(ep.get("metrics"), dict) else {}
        lines = [
            "[SENSORIMOTOR EPISODE]",
            f"Goal: {ep.get('goal') or ''}",
            f"Before: {json.dumps(ep.get('before') or {}, default=str)[:240]}",
            f"Action: {action.get('skill') or action.get('command') or action}",
            f"Actual: {json.dumps(actual, default=str)[:240]}",
            f"After: {json.dumps(ep.get('perception_after') or {}, default=str)[:240]}",
            f"Success: {outcome.get('success')}",
        ]
        if metrics:
            lines.append(f"Metrics: {json.dumps(metrics, default=str)[:200]}")
        return "\n".join(lines)

    # --- spatial ---
    def relate(
        self,
        a: str,
        rel: str,
        b: str,
        *,
        confidence: float = 0.8,
        source: str = "OBSERVATION",
    ) -> dict[str, Any]:
        edge = {
            "a": str(a),
            "rel": str(rel).upper(),
            "b": str(b),
            "confidence": float(confidence),
            "source": str(source),
            "observed_at": _iso(),
            "last_confirmed": _iso(),
        }
        with self._lock:
            rec = _read_json(self._spatial_path(), {"places": {}, "edges": []})
            edges = [e for e in (rec.get("edges") or []) if not (e.get("a") == edge["a"] and e.get("rel") == edge["rel"] and e.get("b") == edge["b"])]
            edges.append(edge)
            rec["edges"] = edges[-80:]
            _write_json(self._spatial_path(), rec)
        return edge

    def spatial_graph(self) -> dict[str, Any] | None:
        path = self._spatial_path()
        if not path.is_file():
            return None
        rec = _read_json(path, None)
        return rec if isinstance(rec, dict) else None

    def render_spatial(self) -> str:
        rec = self.spatial_graph()
        if not rec:
            return ""
        edges = rec.get("edges") or []
        if not edges:
            return ""
        lines = ["[TEELA SPATIAL]"]
        for e in edges[-12:]:
            lines.append(f"{e.get('a')} {e.get('rel')} {e.get('b')} conf={e.get('confidence')}")
        return "\n".join(lines)

    # --- safety ---
    def update_safety(self, fields: dict[str, Any], *, source: str = "MINIOS") -> dict[str, Any]:
        snap = {
            "emergency_stop": bool(fields.get("emergency_stop", False)),
            "motion_allowed": bool(fields.get("motion_allowed", True)),
            "joint_limit_violation": bool(fields.get("joint_limit_violation", False)),
            "collision_risk": bool(fields.get("collision_risk", False)),
            "temperature_fault": bool(fields.get("temperature_fault", False)),
            "actuator_fault": bool(fields.get("actuator_fault", False)),
            "communication_fault": bool(fields.get("communication_fault", False)),
            "human_proximity": fields.get("human_proximity"),
            "unsafe_motion_reason": fields.get("unsafe_motion_reason"),
            "status": str(fields.get("status") or "nominal"),
            "source": str(source),
            "provenance": "RUNTIME",
            "timestamp": _iso(),
        }
        _write_json(self._safety_path(), snap)
        return snap

    def safety_snapshot(self) -> dict[str, Any] | None:
        path = self._safety_path()
        if not path.is_file():
            return None
        rec = _read_json(path, None)
        return rec if isinstance(rec, dict) else None

    def render_safety(self) -> str:
        snap = self.safety_snapshot()
        if not snap:
            return ""
        return (
            "[TEELA SAFETY NOW]\n"
            f"Status: {snap.get('status')}\n"
            f"E-stop: {snap.get('emergency_stop')}\n"
            f"Motion allowed: {snap.get('motion_allowed')}\n"
            f"Source: {snap.get('source')}"
        )

    # --- motor skills ---
    def motor_skill(self, skill_id: str) -> dict[str, Any] | None:
        rec = _read_json(self._motor_path(), {"skills": {}})
        sk = (rec.get("skills") or {}).get(str(skill_id))
        return dict(sk) if isinstance(sk, dict) else None

    def upsert_motor_skill(self, skill_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            rec = _read_json(self._motor_path(), {"skills": {}})
            skills = rec.setdefault("skills", {})
            cur = dict(skills.get(str(skill_id)) or {})
            cur.setdefault("skill", str(skill_id))
            cur.setdefault("procedure", [])
            cur.setdefault("corrections", [])
            cur.setdefault("learned_parameters", {})
            cur.setdefault("proposed_procedure", None)
            cur.setdefault("validated", False)
            cur.update({k: v for k, v in fields.items() if v is not None})
            skills[str(skill_id)] = cur
            rec["skills"] = skills
            _write_json(self._motor_path(), rec)
            return dict(cur)

    def attach_correction(self, skill_id: str, text: str, *, source: str = "USER_CORRECTION") -> dict[str, Any]:
        sk = self.motor_skill(skill_id) or {"skill": skill_id, "corrections": [], "procedure": []}
        corr = list(sk.get("corrections") or [])
        note = {"text": str(text), "source": source, "at": _iso(), "validated": False}
        corr.append(note)
        sk["corrections"] = corr[-12:]
        # User correction is evidence for the procedural note, not a full validated retune.
        proc = list(sk.get("procedure") or [])
        line = str(text).strip()
        if line and line not in proc:
            proc.append(line)
        sk["procedure"] = proc
        sk["last_correction"] = note
        return self.upsert_motor_skill(skill_id, sk)

    def propose_procedure(self, skill_id: str, procedure: list[str], *, source: str = "QWEN_INFERENCE") -> dict[str, Any]:
        return self.upsert_motor_skill(
            skill_id,
            {
                "proposed_procedure": {"steps": list(procedure), "source": source, "at": _iso()},
                "validated": False,
            },
        )

    def validate_from_episode(self, skill_id: str, episode: dict[str, Any]) -> dict[str, Any]:
        outcome = episode.get("outcome") if isinstance(episode.get("outcome"), dict) else {}
        if not outcome.get("success"):
            sk = self.motor_skill(skill_id) or {"skill": skill_id}
            sk["last_failed_episode"] = episode.get("episode_id")
            return self.upsert_motor_skill(skill_id, sk)
        proposed = (self.motor_skill(skill_id) or {}).get("proposed_procedure")
        fields: dict[str, Any] = {
            "validated": True,
            "last_validated": _iso(),
            "last_success_episode": episode.get("episode_id"),
        }
        metrics = episode.get("metrics") if isinstance(episode.get("metrics"), dict) else {}
        params = dict((self.motor_skill(skill_id) or {}).get("learned_parameters") or {})
        if metrics.get("settle_time_ms") is not None:
            params["neck_pan_settle_ms"] = metrics.get("settle_time_ms")
        if metrics.get("final_error_deg") is not None:
            params["acceptable_error_deg"] = max(3, float(metrics.get("final_error_deg") or 3))
        fields["learned_parameters"] = params
        if isinstance(proposed, dict) and proposed.get("steps"):
            fields["procedure"] = list(proposed.get("steps") or [])
            fields["proposed_procedure"] = None
        return self.upsert_motor_skill(skill_id, fields)

    def find_motor_skill(self, query: str) -> dict[str, Any] | None:
        rec = _read_json(self._motor_path(), None)
        if not isinstance(rec, dict):
            return None
        q = (query or "").strip().lower()
        best = None
        for sid, sk in (rec.get("skills") or {}).items():
            blob = f"{sid} {sk.get('skill') or ''} {' '.join(sk.get('procedure') or [])}".lower()
            if q and any(tok in blob for tok in q.replace("-", " ").split() if len(tok) > 3):
                best = dict(sk)
                best.setdefault("skill", sid)
                break
            if sid.replace("_", " ") in q or sid in q.replace(" ", "_"):
                best = dict(sk)
                best.setdefault("skill", sid)
        return best

    def render_motor(self, sk: dict[str, Any]) -> str:
        lines = ["[MOTOR SKILL]", f"Skill: {sk.get('skill')}"]
        if sk.get("procedure"):
            lines.append("Procedure: " + "; ".join(str(x) for x in sk.get("procedure")[:8]))
        if sk.get("corrections"):
            last = sk["corrections"][-1]
            lines.append("Correction: " + str(last.get("text") if isinstance(last, dict) else last))
        lines.append(f"Validated: {bool(sk.get('validated'))}")
        if sk.get("proposed_procedure"):
            lines.append("Proposed (unvalidated): " + str(sk.get("proposed_procedure")))
        if sk.get("learned_parameters"):
            lines.append("Parameters: " + json.dumps(sk.get("learned_parameters"), default=str)[:200])
        return "\n".join(lines)

    def apply_user_correction(self, skill_id: str, text: str) -> dict[str, Any]:
        sk = self.attach_correction(skill_id, text)
        mgr = getattr(self.bot, "memory", None)
        if mgr is not None:
            try:
                mgr.write(
                    f"User correction for {skill_id}: {text}",
                    ["correction", "procedural", skill_id, "USER_CORRECTION"],
                )
            except Exception:
                pass
        return sk

    def body_block(self) -> str:
        if not cprof.has_cap(self.bot, "body_state"):
            return ""
        try:
            bid = str(getattr(self.bot, "id", "") or "teela")
            root = getattr(self.bot, "root", None) or getattr(self.bot, "workspace", None) or "."
            store = bs.store_for(bid, Path(str(root)) / "body_state.sqlite")
            snap = store.snapshot()
            return bs.compact_block(snap)
        except Exception:
            return ""


def for_bot(bot: Any) -> EmbodiedMemory:
    return EmbodiedMemory(bot)
