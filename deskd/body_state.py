#!/usr/bin/env python3
"""Authoritative Teela body state. The browser is a view, not the owner.

Live updates stay in memory. SQLite stores current snapshot + meaningful history.
Simulation mode: MiniOS/virtual overlay is measured truth unless hardware is attached.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EPS = 1.5
_HISTORY_CAP = 200
_stores: dict[str, BodyStateStore] = {}
_stores_lock = threading.Lock()


def store_for(bot_id: str, path: str | Path, *, mode: str = "simulated") -> BodyStateStore:
    key = f"{bot_id}:{path}"
    with _stores_lock:
        cur = _stores.get(key)
        if cur is None:
            cur = BodyStateStore(path, bot_id=bot_id, mode=mode)
            _stores[key] = cur
        return cur

_CREATE = """
CREATE TABLE IF NOT EXISTS body_state_current (
  bot_id TEXT PRIMARY KEY,
  revision INTEGER NOT NULL,
  ts REAL NOT NULL,
  payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS body_state_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  bot_id TEXT NOT NULL,
  revision INTEGER NOT NULL,
  ts REAL NOT NULL,
  payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS body_hist_bot ON body_state_history(bot_id, revision);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _joints_map(raw: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if isinstance(v, dict):
            if v.get("actual") is not None:
                try:
                    out[str(k)] = float(v.get("actual"))
                except (TypeError, ValueError):
                    continue
            continue
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _pack_joints(commanded: dict[str, float], actual: dict[str, float]) -> dict[str, dict[str, float]]:
    names = set(commanded) | set(actual)
    packed: dict[str, dict[str, float]] = {}
    for name in names:
        cmd = commanded.get(name)
        act = actual.get(name, cmd if cmd is not None else 0.0)
        err = 0.0
        if cmd is not None:
            err = float(act) - float(cmd)
        packed[name] = {
            "commanded": float(cmd if cmd is not None else act),
            "actual": float(act),
            "error": float(err),
        }
    return packed


class BodyStateStore:
    def __init__(self, path: str | Path, *, bot_id: str, mode: str = "simulated") -> None:
        self.path = Path(path)
        self.bot_id = str(bot_id or "teela")
        self.mode = "physical" if mode == "physical" else "simulated"
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_CREATE)
        self._conn.commit()
        self._commanded: dict[str, float] = {}
        self._actual: dict[str, float] = {}
        self._pose = "home"
        self._motion = "idle"
        self._waving = False
        self._last_action: str | None = None
        self._source = "simulation"
        self._revision = 0
        self._ts = time.time()
        self._confirmation = "last_known"
        self._load()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def _load(self) -> None:
        row = self._conn.execute(
            "SELECT revision, ts, payload FROM body_state_current WHERE bot_id=?",
            (self.bot_id,),
        ).fetchone()
        if not row:
            return
        rev, ts, raw = row
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        self._revision = int(rev or 0)
        self._ts = float(ts or time.time())
        self._ingest_payload(payload)
        # Disk restore is last-known until live telemetry/command confirms.
        self._confirmation = "last_known"

    def _ingest_payload(self, payload: dict[str, Any]) -> None:
        joints = payload.get("joints") if isinstance(payload.get("joints"), dict) else {}
        commanded: dict[str, float] = {}
        actual: dict[str, float] = {}
        for name, rec in joints.items():
            if isinstance(rec, dict):
                if rec.get("commanded") is not None:
                    try:
                        commanded[str(name)] = float(rec["commanded"])
                    except (TypeError, ValueError):
                        pass
                if rec.get("actual") is not None:
                    try:
                        actual[str(name)] = float(rec["actual"])
                    except (TypeError, ValueError):
                        pass
            else:
                try:
                    actual[str(name)] = float(rec)
                except (TypeError, ValueError):
                    pass
        if commanded:
            self._commanded = commanded
        if actual:
            self._actual = actual
        if payload.get("pose"):
            self._pose = str(payload.get("pose"))
        if payload.get("motion"):
            self._motion = str(payload.get("motion"))
        if "waving" in payload:
            self._waving = bool(payload.get("waving"))
        if payload.get("last_action") is not None:
            self._last_action = str(payload.get("last_action") or "") or None
        if payload.get("source"):
            self._source = str(payload.get("source"))
        if payload.get("mode") in {"physical", "simulated"}:
            self.mode = str(payload.get("mode"))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, Any]:
        joints = _pack_joints(self._commanded, self._actual)
        errors = {
            name: rec["error"]
            for name, rec in joints.items()
            if abs(rec.get("error") or 0) >= _EPS
        }
        reached = not errors
        age_ms = max(0, int((time.time() - self._ts) * 1000))
        return {
            "ok": True,
            "revision": int(self._revision),
            "timestamp": _now_iso(),
            "ts": float(self._ts),
            "source": self._source,
            "mode": self.mode,
            "confirmation": self._confirmation,
            "confirmation_status": self._confirmation,
            "freshness_ms": age_ms,
            "pose": self._pose,
            "motion": self._motion,
            "waving": bool(self._waving),
            "last_action": self._last_action,
            "joints": joints,
            "base": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "target_reached": reached,
            "discrepancies": errors,
        }

    def apply_commanded(
        self,
        joints: dict[str, Any] | None,
        *,
        pose: str | None = None,
        motion: str | None = None,
        waving: bool | None = None,
        last_action: str | None = None,
    ) -> dict[str, Any]:
        packed = _joints_map(joints)
        with self._lock:
            if packed:
                self._commanded.update(packed)
                # Simulation default: command is measured until hardware reports otherwise.
                if self.mode == "simulated":
                    self._actual.update(packed)
            if pose:
                self._pose = str(pose)
            if motion:
                self._motion = str(motion)
            if waving is not None:
                self._waving = bool(waving)
            if last_action is not None:
                self._last_action = str(last_action) or None
            self._source = "command"
            self._confirmation = "current_confirmed"
            self._bump_persist(meaningful=True)
            return self._snapshot_unlocked()

    def apply_measured(
        self,
        joints: dict[str, Any] | None,
        *,
        pose: str | None = None,
        motion: str | None = None,
        waving: bool | None = None,
        source: str = "simulation",
        revision: int | None = None,
    ) -> dict[str, Any]:
        packed = _joints_map(joints)
        with self._lock:
            if revision is not None and int(revision) < int(self._revision):
                return {"accepted": False, "revision": int(self._revision), **self._snapshot_unlocked()}
            changed = False
            for name, val in packed.items():
                prev = self._actual.get(name)
                if prev is None or abs(float(val) - float(prev)) >= _EPS:
                    changed = True
                self._actual[name] = float(val)
            if pose and str(pose) != self._pose:
                self._pose = str(pose)
                changed = True
            if motion and str(motion) != self._motion:
                self._motion = str(motion)
                changed = True
            if waving is not None and bool(waving) != self._waving:
                self._waving = bool(waving)
                changed = True
            self._source = str(source or self._source)
            self._confirmation = "current_confirmed"
            self._bump_persist(meaningful=changed)
            out = self._snapshot_unlocked()
            out["accepted"] = True
            return out

    def _bump_persist(self, *, meaningful: bool) -> None:
        self._ts = time.time()
        if meaningful:
            self._revision = int(self._revision) + 1
        snap = self._snapshot_unlocked()
        blob = json.dumps(snap, default=str)
        self._conn.execute(
            "INSERT INTO body_state_current(bot_id, revision, ts, payload) VALUES(?,?,?,?) "
            "ON CONFLICT(bot_id) DO UPDATE SET revision=excluded.revision, ts=excluded.ts, payload=excluded.payload",
            (self.bot_id, int(self._revision), float(self._ts), blob),
        )
        if meaningful:
            self._conn.execute(
                "INSERT INTO body_state_history(bot_id, revision, ts, payload) VALUES(?,?,?,?)",
                (self.bot_id, int(self._revision), float(self._ts), blob),
            )
            extra = self._conn.execute(
                "SELECT id FROM body_state_history WHERE bot_id=? ORDER BY id DESC",
                (self.bot_id,),
            ).fetchall()
            if len(extra) > _HISTORY_CAP:
                cutoff = extra[_HISTORY_CAP][0]
                self._conn.execute(
                    "DELETE FROM body_state_history WHERE bot_id=? AND id<=?",
                    (self.bot_id, cutoff),
                )
        self._conn.commit()

    def history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        n = max(1, min(int(limit or 20), _HISTORY_CAP))
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM body_state_history WHERE bot_id=? ORDER BY revision DESC LIMIT ?",
                (self.bot_id, n),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for (raw,) in rows:
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
        return out

    def joint(self, name: str) -> dict[str, Any]:
        snap = self.snapshot()
        rec = (snap.get("joints") or {}).get(str(name))
        if not isinstance(rec, dict):
            return {"ok": False, "error": f"unknown joint {name}", "revision": snap.get("revision")}
        return {"ok": True, "joint": str(name), **rec, "revision": snap.get("revision")}


def compact_block(snap: dict[str, Any] | None) -> str:
    st = snap if isinstance(snap, dict) else {}
    joints = st.get("joints") if isinstance(st.get("joints"), dict) else {}

    def deg(name: str) -> float:
        rec = joints.get(name) if isinstance(joints.get(name), dict) else {}
        try:
            return float(rec.get("actual") or 0)
        except (TypeError, ValueError):
            return 0.0

    r_sh, r_el = deg("right_shoulder"), deg("right_elbow")
    l_sh, l_el = deg("left_shoulder"), deg("left_elbow")
    pan = deg("neck_pan")
    right = "raised" if r_sh >= 120 else ("wave-hold" if r_el >= 90 and r_sh >= 15 else "resting")
    left = "raised" if l_sh >= 120 else ("wave-hold" if l_el >= 90 and l_sh >= 15 else "resting")
    disc = st.get("discrepancies") if isinstance(st.get("discrepancies"), dict) else {}
    disc_line = "none"
    if disc:
        bits = []
        for name, err in list(disc.items())[:6]:
            rec = joints.get(name) if isinstance(joints.get(name), dict) else {}
            bits.append(
                f"{name} commanded {float(rec.get('commanded') or 0):.0f}°, actual {float(rec.get('actual') or 0):.0f}°"
            )
        disc_line = "; ".join(bits)
    age = st.get("freshness_ms")
    confirmation = st.get("confirmation") or st.get("confirmation_status") or "last_known"
    return (
        "[TEELA BODY NOW]\n"
        f"Mode: {st.get('mode') or 'simulated'}\n"
        f"Revision: {st.get('revision') or 0}\n"
        f"Confirmation: {confirmation}\n"
        f"Authority: MiniOS/body-state (overrides conversation)\n"
        f"State age: {age if age is not None else '?'} ms\n"
        f"Motion: {st.get('motion') or 'idle'} pose={st.get('pose') or 'home'} waving={bool(st.get('waving'))}\n"
        f"Head: yaw {pan:.0f}°\n"
        f"Right arm: shoulder {r_sh:.0f}° elbow {r_el:.0f}° ({right})\n"
        f"Left arm: shoulder {l_sh:.0f}° elbow {l_el:.0f}° ({left})\n"
        f"Last action: {st.get('last_action') or 'none'}\n"
        f"Discrepancies: {disc_line}\n"
        "This is measured BodyState, not conversational memory."
    )


def overlay_from_snapshot(snap: dict[str, Any] | None) -> dict[str, Any]:
    st = snap if isinstance(snap, dict) else {}
    joints = _joints_map(st.get("joints"))
    return {
        "mode": "virtual",
        "pose": st.get("pose") or "home",
        "motion": st.get("motion") or "idle",
        "waving": bool(st.get("waving")),
        "joints": dict(joints),
        "live": dict(joints),
        "stamp_t": time.time(),
        "revision": st.get("revision"),
        "source": "body_state",
    }


def restore_overlay(bot_id: str, snap: dict[str, Any] | None) -> dict[str, Any]:
    """Put last authoritative pose into the overlay so a browser reconnect cannot zero it."""
    import virtual_body

    body = overlay_from_snapshot(snap)
    virtual_body.note_state(str(bot_id or ""), body)
    return body
