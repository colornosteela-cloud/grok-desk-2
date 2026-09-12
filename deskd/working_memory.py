"""Read-only Working Memory Manifest.

A monitoring layer in front of the current assembler. UI and HTTP talk only
to this module. It does not change retrieval ranking, compaction thresholds,
or the context window.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import telemetry as tel

BODY_STALE_MS = 5000
HISTORY_MAX = 200
EVENTS_MAX = 200
ITEMS_MAX = 400
DEFAULT_ITEM_LIMIT = 100
DEFAULT_MEMORY_LIMIT = 50

PRESSURE_GREEN = "GREEN"
PRESSURE_YELLOW = "YELLOW"
PRESSURE_ORANGE = "ORANGE"
PRESSURE_RED = "RED"
PRESSURE_CRITICAL = "CRITICAL"

# GREEN if utilization < 45%; YELLOW from 45% inclusive so one state is unambiguous.
PRESSURE_BANDS = (
    (0.45, PRESSURE_GREEN),
    (0.65, PRESSURE_YELLOW),
    (0.80, PRESSURE_ORANGE),
    (0.90, PRESSURE_RED),
)

SECTION_KEYS = (
    "core",
    "body_state",
    "safety_state",
    "world_state",
    "conversation",
    "active_task",
    "retrieved_memory",
    "skills",
    "perception",
    "sensorimotor",
    "tools",
    "workspace",
    "other",
    "unknown",
)

REASON_CODES = (
    "PINNED_CORE",
    "CURRENT_BODY_STATE",
    "CURRENT_WORLD_STATE",
    "ACTIVE_TASK",
    "ACTIVE_GOAL",
    "RECENT_CONVERSATION",
    "SEMANTIC_MATCH",
    "ENTITY_MATCH",
    "SKILL_MATCH",
    "USER_CORRECTION",
    "UNRESOLVED_TASK",
    "TEMPORAL_MATCH",
    "EXPLICIT_RECALL",
    "SAFETY_RELEVANT",
    "PERCEPTION_RELEVANT",
    "TASK_CHECKPOINT",
    "SYSTEM_REQUIRED",
)

REASON_LABELS = {
    "PINNED_CORE": "this item is pinned core identity",
    "CURRENT_BODY_STATE": "it is the current authoritative body state",
    "CURRENT_WORLD_STATE": "it is the current world state",
    "ACTIVE_TASK": "it is relevant to the active task",
    "ACTIVE_GOAL": "it is the pinned active goal",
    "RECENT_CONVERSATION": "it is recent conversation",
    "SEMANTIC_MATCH": "it is a strong semantic match to the current request",
    "ENTITY_MATCH": "it matches an entity in the current request",
    "SKILL_MATCH": "it matches the current skill",
    "USER_CORRECTION": "it contains a previous user correction",
    "UNRESOLVED_TASK": "it belongs to an unresolved task",
    "TEMPORAL_MATCH": "it matches the current time window",
    "EXPLICIT_RECALL": "it was explicitly recalled",
    "SAFETY_RELEVANT": "it is safety-relevant",
    "PERCEPTION_RELEVANT": "it is a current perception event",
    "TASK_CHECKPOINT": "it is a task checkpoint",
    "SYSTEM_REQUIRED": "it is required system context",
}

NAMED_SOURCES = (
    "system",
    "user",
    "assistant",
    "conversation",
    "conversation_summary",
    "body_state",
    "world_state",
    "active_task",
    "episodic_memory",
    "semantic_memory",
    "procedural_memory",
    "self_model",
    "skill",
    "perception",
    "sensorimotor",
    "safety_state",
    "MiniOS",
    "tool_result",
    "agent_result",
    "external_document",
    "unknown",
)

EVENT_TYPES = (
    "MEMORY_RETRIEVED",
    "MEMORY_EVICTED",
    "CONTEXT_COMPACTED",
    "CONTEXT_REBUILT",
    "CONTEXT_ITEM_LOADED",
    "CONTEXT_ITEM_EVICTED",
    "TASK_CHECKPOINT_LOADED",
    "TASK_CHECKPOINT_SAVED",
    "BODY_STATE_UPDATED",
    "WORLD_STATE_UPDATED",
    "PERCEPTION_EVENT",
    "SENSORIMOTOR_RECORDED",
    "SKILL_LOADED",
    "CORRECTION_LOADED",
    "CONVERSATION_SUMMARIZED",
    "CONTEXT_PRESSURE_CHANGED",
)

MEMORY_TYPES = (
    "episodic",
    "semantic",
    "procedural",
    "relational",
    "self",
    "task",
    "unknown",
)

_TYPE_TAGS = {
    "episodic": ("episodic", "episode", "experience", "event"),
    "semantic": ("semantic", "knowledge", "fact"),
    "procedural": ("procedural", "procedure", "skill", "how-to"),
    "relational": ("relational", "person", "relationship", "people"),
    "self": ("self", "identity", "self-model"),
    "task": ("task", "goal", "checkpoint", "horizon"),
}

_SOURCE_FOR_TYPE = {
    "episodic": "episodic_memory",
    "semantic": "semantic_memory",
    "procedural": "procedural_memory",
    "self": "self_model",
    "task": "active_task",
}

_SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "id_token",
    "client_secret",
    "private_key",
    "password",
    "passwd",
    "secret",
    "token",
    "cookie",
    "bearer",
    "credentials",
    "auth",
}

_FORBIDDEN_KEYS = {
    "thought",
    "reasoning",
    "scratchpad",
    "chain_of_thought",
    "chainofthought",
    "hidden_reasoning",
    "private_reasoning",
    "thinking",
    "model_thought",
    "internal_thought",
    "hidden_thought",
    "private_scratchpad",
}

_REDACTED = "[REDACTED]"
_UNAVAILABLE = "unavailable"

_SECRET_IN_TEXT = re.compile(
    r"(?i)(api[_-]?key|password|secret|authorization|access[_-]?token|bearer|cookie)\s*[:=]\s*([^\s,;]{4,})"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-+=/]{8,}")

_ledgers: dict[str, "WorkingMemoryLedger"] = {}
_ledgers_lock = threading.Lock()


def utc_iso(ts: float | None = None) -> str:
    t = datetime.fromtimestamp(ts if ts is not None else time.time(), tz=timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def pressure_from_utilization(u: float | None) -> str | None:
    """Map capacity utilization to a pressure band. None if utilization is unknown."""
    if u is None:
        return None
    try:
        x = float(u)
    except (TypeError, ValueError):
        return None
    if x != x or x < 0:  # noqa: PLR0124 NaN
        return None
    for upper, name in PRESSURE_BANDS:
        if x < upper:
            return name
    return PRESSURE_CRITICAL


def occupancy_stats(used: int | None, capacity: int | None) -> dict[str, Any]:
    cap = _pos_int(capacity)
    used_n = _nonneg_int(used)
    out: dict[str, Any] = {
        "capacity_tokens": cap,
        "used_tokens": used_n,
        "free_tokens": None,
        "utilization": None,
        "capacity_percent": None,
        "pressure": None,
    }
    if cap is None:
        return out
    if used_n is None:
        return out
    free = max(0, cap - used_n)
    util = used_n / cap if cap else None
    out["free_tokens"] = free
    out["utilization"] = util
    out["capacity_percent"] = None if util is None else round(util * 100.0, 4)
    out["pressure"] = pressure_from_utilization(util)
    return out


def occupied_percents(sections: dict[str, int] | None) -> dict[str, float]:
    """Percent of currently occupied tokens, not of capacity."""
    secs = sections or {}
    total = 0
    clean: dict[str, int] = {}
    for k, v in secs.items():
        n = _nonneg_int(v)
        if n is None:
            continue
        clean[str(k)] = n
        total += n
    if total <= 0:
        return {}
    return {k: (n / total) * 100.0 for k, n in clean.items()}


def count_section_tokens(text: str) -> tuple[int, bool]:
    """Local tokenizer count for a section/item. Never characters/4.

    Always estimated=True: this is not llama.cpp/runtime occupancy.
    """
    n = tel.count_tokens_local(text or "")
    return int(n), True


def score_from_bm25(rank: float | None) -> float | None:
    if rank is None:
        return None
    try:
        x = float(rank)
    except (TypeError, ValueError):
        return None
    if x != x:  # noqa: PLR0124
        return None
    # FTS5 bm25 is typically negative (more negative = better). Logistic in [0,1].
    try:
        s = 1.0 / (1.0 + math.exp(x))
    except OverflowError:
        s = 0.0 if x > 0 else 1.0
    return max(0.0, min(1.0, float(s)))


def classify_memory_type(tags: list[str] | None, text: str = "") -> str:
    tset = {str(t).strip().lower() for t in (tags or []) if str(t).strip()}
    if not tset:
        blob = (text or "").lower()
        if "correction" in blob:
            return "procedural"
        return "unknown"
    if "correction" in tset or "corrections" in tset:
        return "procedural"
    for kind, keys in _TYPE_TAGS.items():
        if tset & set(keys):
            return kind
    return "unknown"


def source_for_memory_type(kind: str) -> str:
    return _SOURCE_FOR_TYPE.get(kind, "unknown")


def why_loaded(codes: list[str] | None) -> str:
    labels: list[str] = []
    for c in codes or []:
        key = str(c or "").strip().upper()
        if not key:
            continue
        labels.append(REASON_LABELS.get(key, key.replace("_", " ").lower()))
    if not labels:
        return "Unknown — runtime did not record a retrieval reason."
    if len(labels) == 1:
        s = labels[0]
        return s[0].upper() + s[1:] + "."
    return "Relevant because " + ", ".join(labels[:-1]) + ", and " + labels[-1] + "."


def unavailable(reason: str) -> dict[str, Any]:
    return {"status": _UNAVAILABLE, "reason": reason}


def is_unavailable(value: Any) -> bool:
    return isinstance(value, dict) and value.get("status") == _UNAVAILABLE


def occupancy_used(bot: Any) -> int | None:
    source = str(getattr(bot, "context_source", "") or "")
    if not source:
        return None
    return _nonneg_int(getattr(bot, "context_used", None))


def occupancy_capacity(bot: Any) -> int | None:
    n = _pos_int(getattr(bot, "context_window", None))
    if n:
        return n
    model = str(getattr(bot, "model", "") or "")
    return _pos_int(tel.resolve_context_window(model)) or None


def used_is_estimated(bot: Any) -> bool:
    src = str(getattr(bot, "context_source", "") or "")
    return src in {"", "tokenizer"}


def redact_secrets(value: Any) -> Any:
    """Drop private CoT keys and redact known secret fields. Recursive."""
    return _redact(value, ancestors=())


def strip_forbidden(value: Any) -> Any:
    return redact_secrets(value)


class WorkingMemoryLedger:
    def __init__(self, bot_id: str) -> None:
        self.bot_id = bot_id
        self.lock = threading.RLock()
        self.items: list[dict[str, Any]] = []
        self.sections: dict[str, int] = {}
        self.section_estimated: bool = True
        self.loaded_ids: set[str] = set()
        self.retrieval: dict[str, Any] | None = None
        self.history: deque[dict[str, Any]] = deque(maxlen=HISTORY_MAX)
        self.events: deque[dict[str, Any]] = deque(maxlen=EVENTS_MAX)
        self.last_used: int | None = None
        self.last_pressure: str | None = None
        self.last_updated: str | None = None
        self.last_body_revision: int | None = None
        self.last_body_ts: float | None = None
        self.assembly_at: str | None = None
        self.just_compacted: bool = False
        self.just_retrieved: bool = False
        self.last_hist_ts: float = 0.0


def ledger_for(bot_id: str) -> WorkingMemoryLedger:
    key = str(bot_id or "")
    with _ledgers_lock:
        cur = _ledgers.get(key)
        if cur is None:
            cur = WorkingMemoryLedger(key)
            _ledgers[key] = cur
        return cur


def note_occupancy(bot: Any) -> dict[str, Any]:
    """High-frequency path: occupancy fields only. No tokenize, no DB scan."""
    used = occupancy_used(bot)
    cap = occupancy_capacity(bot)
    stats = occupancy_stats(used, cap)
    now = time.time()
    iso = utc_iso(now)
    extra: dict[str, Any] = {
        "pressure": stats["pressure"],
        "utilization": stats["utilization"],
        "free_tokens": stats["free_tokens"],
        "last_updated": iso,
    }
    if used is None:
        extra["used_status"] = _UNAVAILABLE
    led = ledger_for(str(getattr(bot, "id", "") or ""))
    with led.lock:
        led.last_updated = iso
        prev_used = led.last_used
        prev_p = led.last_pressure
        pressure = stats["pressure"]
        if pressure and prev_p and pressure != prev_p:
            _push_event(
                led,
                "CONTEXT_PRESSURE_CHANGED",
                before=prev_p,
                after=pressure,
                used_tokens=used,
                capacity_tokens=cap,
            )
        compact = bool(led.just_compacted)
        retrieved = bool(led.just_retrieved)
        should_hist = used is not None and (
            prev_used is None or used != prev_used or (now - led.last_hist_ts) >= 0.5
        )
        if should_hist and used is not None:
            added = None if prev_used is None else max(0, int(used) - int(prev_used))
            removed = None if prev_used is None else max(0, int(prev_used) - int(used))
            led.history.append(
                {
                    "timestamp": iso,
                    "tokens_used": used,
                    "percentage": stats["capacity_percent"],
                    "pressure": pressure,
                    "tokens_added": added,
                    "tokens_removed": removed,
                    "compaction": compact,
                    "retrieval": retrieved,
                }
            )
            led.last_hist_ts = now
        led.last_used = used
        led.last_pressure = pressure
        led.just_compacted = False
        led.just_retrieved = False
    return extra


def ingest_manager_events(bot: Any) -> None:
    mgr = getattr(bot, "memory", None)
    events = getattr(mgr, "context_events", None) if mgr is not None else None
    if not events:
        return
    led = ledger_for(str(getattr(bot, "id", "") or ""))
    with led.lock:
        while True:
            try:
                ev = events.popleft()
            except (IndexError, AttributeError):
                break
            if not isinstance(ev, dict):
                continue
            kind = str(ev.get("type") or "")
            if kind == "CONTEXT_COMPACTED":
                led.just_compacted = True
            if kind == "MEMORY_RETRIEVED":
                led.just_retrieved = True
            _push_event(led, kind, **{k: v for k, v in ev.items() if k != "type"})


def capture_turn(bot: Any, payload: dict[str, Any] | None, assembled: Any = None) -> None:
    """Record last-turn section/item/retrieval metadata. Called on assemble, not the live tick."""
    ingest_manager_events(bot)
    bid = str(getattr(bot, "id", "") or "")
    led = ledger_for(bid)
    now = utc_iso()
    items: list[dict[str, Any]] = []
    sections: dict[str, int] = {k: 0 for k in SECTION_KEYS}

    retrieval = None
    assembled_sections: set[str] = set()
    if assembled is not None:
        retrieval = getattr(assembled, "retrieval", None)
        for it in list(getattr(assembled, "items", None) or []):
            if isinstance(it, dict):
                items.append(dict(it))
                if it.get("section"):
                    assembled_sections.add(str(it.get("section")))

    payload = payload if isinstance(payload, dict) else {}
    messages = list(payload.get("messages") or [])
    parsed_items, parsed_sections = _items_from_payload(messages, assembled, now)
    by_id = {str(it.get("id") or ""): it for it in items if it.get("id")}
    skip_dup = {"retrieved_memory", "conversation", "active_task"} & assembled_sections
    for it in parsed_items:
        iid = str(it.get("id") or "")
        if str(it.get("section") or "") in skip_dup and iid.startswith("sys_"):
            continue
        if iid and iid in by_id:
            by_id[iid].update({k: v for k, v in it.items() if v is not None})
        else:
            items.append(it)
            if iid:
                by_id[iid] = it
    _ = parsed_sections  # section sums are rebuilt from items below

    tools = payload.get("tools")
    if tools:
        blob = json.dumps(tools, default=str)
        n, est = count_section_tokens(blob)
        items.append(
            _item(
                id="tools_specs",
                section="tools",
                type="unknown",
                title="Tool / capability specifications",
                token_count=n,
                estimated=est,
                source="system",
                reason_codes=["SYSTEM_REQUIRED"],
                loaded_at=now,
                pinned=False,
            )
        )
        sections["tools"] = sections.get("tools", 0) + n

    # Recompute section sums from items so they stay consistent.
    summed: dict[str, int] = {k: 0 for k in SECTION_KEYS}
    loaded: set[str] = set()
    for it in items:
        sec = str(it.get("section") or "unknown")
        if sec not in summed:
            sec = "unknown"
            it["section"] = sec
        n = _nonneg_int(it.get("token_count")) or 0
        summed[sec] += n
        iid = str(it.get("id") or "")
        if iid:
            loaded.add(iid)
        it.setdefault("why_loaded", why_loaded(it.get("reason_codes") or []))
        it.setdefault("in_context", True)

    if retrieval and isinstance(retrieval, dict):
        retrieval = dict(retrieval)
        retrieval.setdefault("timestamp", now)
        selected = retrieval.get("selected") or []
        if selected:
            _push_event_unlocked_safe(
                led,
                "MEMORY_RETRIEVED",
                query_intent=retrieval.get("query_intent") or retrieval.get("query"),
                selected=len(selected),
                candidates_examined=retrieval.get("candidates_examined"),
                injected_tokens=retrieval.get("injected_tokens"),
            )

    with led.lock:
        led.items = [_sanitize_item(it) for it in items][:ITEMS_MAX]
        led.sections = {k: int(summed.get(k) or 0) for k in SECTION_KEYS}
        led.section_estimated = True
        led.loaded_ids = loaded
        led.retrieval = redact_secrets(retrieval) if retrieval else led.retrieval
        led.assembly_at = now
        led.last_updated = now
        selected = (retrieval or {}).get("selected") or [] if isinstance(retrieval, dict) else []
        if selected:
            led.just_retrieved = True

    _capture_body_event(bot, led)
    _capture_task_event(bot, led)


def build_summary(bot: Any) -> dict[str, Any]:
    ingest_manager_events(bot)
    used = occupancy_used(bot)
    cap = occupancy_capacity(bot)
    stats = occupancy_stats(used, cap)
    led = ledger_for(str(getattr(bot, "id", "") or ""))
    now = utc_iso()
    with led.lock:
        sections = dict(led.sections)
        section_est = bool(led.section_estimated)
        loaded_n = len(led.loaded_ids)
        last_updated = led.last_updated or now
        assembly_at = led.assembly_at
        items_snapshot = list(led.items)
    section_out: dict[str, int | None] = {}
    embodied = True
    try:
        import cognitive_profile as _cprof

        embodied = _cprof.is_embodied(bot)
    except Exception:
        embodied = True
    embodied_sections = {"body_state", "world_state", "perception", "safety_state", "sensorimotor"}
    for k in SECTION_KEYS:
        if not assembly_at:
            section_out[k] = None
        elif k in embodied_sections and not embodied:
            # Build bots omit robot sections entirely — never a fake zero.
            section_out[k] = None
        elif k in {"world_state", "perception", "safety_state", "sensorimotor"} and not int(sections.get(k) or 0):
            # No world/perception objects in the current assembler — not a measured zero.
            section_out[k] = None
        else:
            section_out[k] = int(sections.get(k) or 0)
    occupied = occupied_percents({k: v for k, v in section_out.items() if v})
    stored_total, type_counts, mem_status = _memory_counts(bot)
    currently_loaded = _currently_loaded_memories(items_snapshot)
    body = _body_section(bot)
    world, perception, safety, sensorimotor = _embodied_sections(bot)
    kv = unavailable("Runtime does not expose this metric")
    active_task = _active_task_section(bot, sections)
    runtime = _runtime_section(bot, stats)
    corrections = _corrections_section(bot, items_snapshot)
    skills = _skills_section(bot, items_snapshot)

    used_est = bool(used is not None and used_is_estimated(bot))
    out: dict[str, Any] = {
        "capacity_tokens": stats["capacity_tokens"],
        "used_tokens": stats["used_tokens"],
        "free_tokens": stats["free_tokens"],
        "utilization": stats["utilization"],
        "capacity_percent": stats["capacity_percent"],
        "pressure": stats["pressure"],
        "used_estimated": used_est if stats["used_tokens"] is not None else None,
        "used_status": None if stats["used_tokens"] is not None else _UNAVAILABLE,
        "capacity_status": None if stats["capacity_tokens"] is not None else _UNAVAILABLE,
        "pressure_status": None if stats["pressure"] is not None else _UNAVAILABLE,
        "context_source": str(getattr(bot, "context_source", "") or "") or None,
        "sections": section_out,
        "sections_status": None if assembly_at else _UNAVAILABLE,
        "sections_estimated": section_est if assembly_at else None,
        "occupied_percents": occupied if assembly_at else {},
        "memory": {
            "stored_total": stored_total,
            "currently_loaded": currently_loaded,
            "loaded_context_items": loaded_n,
            "types": type_counts,
            "status": mem_status,
        },
        "active_now_tokens": stats["used_tokens"],
        "stored": {
            "label": "STORED / AVAILABLE FOR RECALL",
            "total": stored_total,
            "currently_loaded": currently_loaded,
        },
        "cognitive_profile": None,
        "active_task": active_task,
        "body_state": body,
        "world_state": world,
        "perception": perception,
        "safety_state": safety,
        "sensorimotor": sensorimotor,
        "kv": kv,
        "runtime": runtime,
        "corrections": corrections,
        "skills": skills,
        "last_updated": last_updated,
        "assembly_at": assembly_at,
        "model": str(getattr(bot, "model", "") or "") or None,
    }
    try:
        import cognitive_profile as _cprof

        out["cognitive_profile"] = _cprof.profile_name(bot)
        out["capabilities"] = _cprof.capabilities(bot)
    except Exception:
        pass
    return redact_secrets(out)


def build_items(
    bot: Any,
    *,
    type: str | None = None,  # noqa: A002
    section: str | None = None,
    pinned: bool | None = None,
    source: str | None = None,
    limit: int = DEFAULT_ITEM_LIMIT,
    offset: int = 0,
    q: str | None = None,
) -> dict[str, Any]:
    led = ledger_for(str(getattr(bot, "id", "") or ""))
    with led.lock:
        rows = list(led.items)
        assembly_at = led.assembly_at
    if not assembly_at:
        return {
            "items": [],
            "total": 0,
            "status": _UNAVAILABLE,
            "reason": "No assembled context has been recorded yet",
        }
    needle = (q or "").strip().lower()
    out: list[dict[str, Any]] = []
    for it in rows:
        if section and str(it.get("section") or "") != section:
            continue
        if type and str(it.get("type") or "") != type:
            continue
        if source and str(it.get("source") or "") != source:
            continue
        if pinned is not None and bool(it.get("pinned")) != bool(pinned):
            continue
        if needle:
            blob = " ".join(
                str(it.get(k) or "")
                for k in ("id", "title", "type", "section", "source", "why_loaded")
            ).lower()
            if needle not in blob:
                continue
        row = dict(it)
        row["why_loaded"] = why_loaded(row.get("reason_codes") or [])
        row["in_context"] = True
        out.append(row)
    limit_n = max(1, min(int(limit or DEFAULT_ITEM_LIMIT), 200))
    offset_n = max(0, int(offset or 0))
    return redact_secrets(
        {
            "items": out[offset_n : offset_n + limit_n],
            "total": len(out),
            "offset": offset_n,
            "limit": limit_n,
        }
    )


def build_memories(
    bot: Any,
    *,
    type: str | None = None,  # noqa: A002
    q: str | None = None,
    offset: int = 0,
    limit: int = DEFAULT_MEMORY_LIMIT,
    corrections: bool = False,
) -> dict[str, Any]:
    mgr = getattr(bot, "memory", None)
    if mgr is None:
        return {"records": [], "total": 0, "status": _UNAVAILABLE, "reason": "Memory backend: Disconnected"}
    store = getattr(mgr, "store", None)
    if store is None:
        return {"records": [], "total": 0, "status": _UNAVAILABLE, "reason": "Memory backend: Disconnected"}
    led = ledger_for(str(getattr(bot, "id", "") or ""))
    with led.lock:
        loaded = set(led.loaded_ids)
    limit_n = max(1, min(int(limit or DEFAULT_MEMORY_LIMIT), 100))
    offset_n = max(0, int(offset or 0))
    try:
        rows, total = store.list_facts(
            offset=offset_n,
            limit=limit_n,
            query=q or "",
            type_filter=type or "",
            corrections=bool(corrections),
        )
    except Exception as e:
        return {
            "records": [],
            "total": 0,
            "status": _UNAVAILABLE,
            "reason": f"Memory backend: {e}",
        }
    records = []
    for rec in rows:
        rid = f"mem_{rec.id}"
        kind = classify_memory_type(rec.tags, rec.text)
        in_ctx = rid in loaded or str(rec.id) in loaded
        records.append(
            redact_secrets(
                {
                    "id": rid,
                    "record": rid,
                    "store": "facts.sqlite",
                    "type": kind,
                    "title": _title(rec.text),
                    "tags": list(rec.tags or []),
                    "created_at": _ts_iso(rec.created_at),
                    "currently_in_context": bool(in_ctx),
                    "presence": "IN CONTEXT" if in_ctx else "STORED ONLY",
                    "source": source_for_memory_type(kind),
                    "media_type": rec.media_type.value if rec.media_type else None,
                }
            )
        )
    types = {}
    try:
        types = store.type_counts()
    except Exception:
        types = {}
    return {
        "records": records,
        "total": int(total),
        "offset": offset_n,
        "limit": limit_n,
        "types": types,
        "stored_total": types.get("total", total),
        "currently_loaded": _currently_loaded_memories_from_ids(loaded),
    }


def build_memory_detail(bot: Any, mem_id: str) -> dict[str, Any]:
    mgr = getattr(bot, "memory", None)
    if mgr is None:
        return {"status": _UNAVAILABLE, "reason": "Memory backend: Disconnected"}
    nid = _parse_mem_id(mem_id)
    if nid is None:
        return {"error": "not found", "status": "not_found"}
    rec = mgr.store.get(nid)
    if rec is None:
        return {"error": "not found", "status": "not_found"}
    led = ledger_for(str(getattr(bot, "id", "") or ""))
    with led.lock:
        loaded = set(led.loaded_ids)
        last_ret = None
        if led.retrieval:
            for row in list(led.retrieval.get("selected") or []) + list(led.retrieval.get("rejected") or []):
                if str(row.get("id") or "") in {f"mem_{nid}", str(nid)}:
                    last_ret = row
                    break
        item = next((it for it in led.items if str(it.get("id") or "") in {f"mem_{nid}", str(nid)}), None)
    kind = classify_memory_type(rec.tags, rec.text)
    in_ctx = f"mem_{nid}" in loaded or str(nid) in loaded
    content = redact_secrets(_clip(rec.text, 4000))
    return redact_secrets(
        {
            "id": f"mem_{nid}",
            "record": f"mem_{nid}",
            "store": "facts.sqlite",
            "type": kind.upper() if kind != "unknown" else "UNKNOWN",
            "created": _ts_iso(rec.created_at),
            "updated": _ts_iso(rec.created_at),
            "importance": None,
            "confidence": (last_ret or {}).get("relevance_score") or (item or {}).get("confidence"),
            "status": "ACTIVE",
            "currently_in_context": bool(in_ctx),
            "presence": "IN CONTEXT" if in_ctx else "STORED ONLY",
            "associated": _associated(rec.tags),
            "tags": list(rec.tags or []),
            "content": content,
            "provenance": "user correction" if "correction" in {t.lower() for t in rec.tags or []} else "persistent memory",
            "last_retrieved": (item or {}).get("loaded_at") or (item or {}).get("last_used"),
            "reason_codes": (item or {}).get("reason_codes") or [],
            "why_loaded": why_loaded((item or {}).get("reason_codes") or []) if in_ctx else None,
            "media_path": rec.media_path,
            "media_type": rec.media_type.value if rec.media_type else None,
        }
    )


def build_retrieval(bot: Any) -> dict[str, Any]:
    led = ledger_for(str(getattr(bot, "id", "") or ""))
    with led.lock:
        ret = dict(led.retrieval) if led.retrieval else None
        events = [e for e in list(led.events) if e.get("type") == "MEMORY_RETRIEVED"]
    if not ret:
        return {
            "status": _UNAVAILABLE,
            "reason": "No retrieval has been recorded yet",
            "events": redact_secrets(events[-20:]),
        }
    return redact_secrets({"status": "ok", **ret, "events": events[-20:]})


def build_history(bot: Any) -> dict[str, Any]:
    led = ledger_for(str(getattr(bot, "id", "") or ""))
    with led.lock:
        hist = list(led.history)
        events = list(led.events)
    return redact_secrets({"history": hist, "events": events})


def build_search(bot: Any, q: str, *, scope: str = "all", limit: int = 40) -> dict[str, Any]:
    needle = (q or "").strip()
    if not needle:
        return {"query": q, "results": []}
    limit_n = max(1, min(int(limit or 40), 100))
    scope_s = (scope or "all").lower()
    results: list[dict[str, Any]] = []
    if scope_s in {"all", "context", "active"}:
        items = build_items(bot, q=needle, limit=limit_n).get("items") or []
        for it in items:
            results.append(
                {
                    "id": it.get("id"),
                    "title": it.get("title"),
                    "type": it.get("type"),
                    "section": it.get("section"),
                    "presence": "IN CONTEXT",
                    "in_context": True,
                    "source": it.get("source"),
                }
            )
    if scope_s in {"all", "memory", "long-term", "skills", "corrections"}:
        corr = scope_s == "corrections"
        typ = "procedural" if scope_s == "skills" else None
        mem = build_memories(bot, q=needle, limit=limit_n, type=typ, corrections=corr)
        for rec in mem.get("records") or []:
            if any(r.get("id") == rec.get("id") for r in results):
                # already listed from active context; keep IN CONTEXT
                continue
            results.append(
                {
                    "id": rec.get("id"),
                    "title": rec.get("title"),
                    "type": rec.get("type"),
                    "section": "retrieved_memory" if rec.get("currently_in_context") else None,
                    "presence": rec.get("presence"),
                    "in_context": bool(rec.get("currently_in_context")),
                    "source": rec.get("source"),
                }
            )
    if scope_s in {"all", "skills", "corrections"}:
        for row in (_skills_search(bot, needle, corrections=(scope_s == "corrections"))):
            if any(r.get("id") == row.get("id") for r in results):
                continue
            results.append(row)
    return redact_secrets({"query": needle, "results": results[:limit_n]})


def build_snapshot(bot: Any) -> dict[str, Any]:
    summary = build_summary(bot)
    items = build_items(bot, limit=200)
    hist = build_history(bot)
    retrieval = build_retrieval(bot)
    snap = {
        "timestamp": utc_iso(),
        "model": summary.get("model"),
        "context_capacity": summary.get("capacity_tokens"),
        "context_utilization": summary.get("utilization"),
        "pressure": summary.get("pressure"),
        "section_token_counts": summary.get("sections"),
        "loaded_context_item_metadata": items.get("items") or [],
        "retrieval_events": (hist.get("events") or []) if isinstance(hist, dict) else [],
        "retrieval": retrieval if retrieval.get("status") != _UNAVAILABLE else retrieval,
        "active_task_metadata": summary.get("active_task"),
        "body_state_revision": (summary.get("body_state") or {}).get("revision")
        if isinstance(summary.get("body_state"), dict)
        else None,
        "world_state_revision": None,
        "memory_counts": summary.get("memory"),
        "kv": summary.get("kv"),
        "runtime": summary.get("runtime"),
        "history": (hist.get("history") or [])[-50:],
    }
    return redact_secrets(snap)


def handle_get(bot: Any, rest: str, qs: dict[str, list[str]]) -> dict[str, Any]:
    """Dispatch a working-memory GET. rest is '' / items / memories / ..."""
    action = (rest or "").strip("/")
    def q(name: str, default: str = "") -> str:
        vals = qs.get(name) or [default]
        return str(vals[0] if vals else default)

    def q_int(name: str, default: int) -> int:
        raw = q(name, str(default))
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    def q_bool(name: str) -> bool | None:
        raw = q(name, "").strip().lower()
        if raw in {"1", "true", "yes"}:
            return True
        if raw in {"0", "false", "no"}:
            return False
        return None

    if action in {"", "summary", "context"}:
        return build_summary(bot)
    if action == "items":
        return build_items(
            bot,
            type=q("type") or None,
            section=q("section") or None,
            pinned=q_bool("pinned"),
            source=q("source") or None,
            limit=q_int("limit", DEFAULT_ITEM_LIMIT),
            offset=q_int("offset", 0),
            q=q("q") or q("query") or None,
        )
    if action.startswith("memories/") and len(action.split("/")) == 2:
        return build_memory_detail(bot, action.split("/", 1)[1])
    if action == "memories":
        return build_memories(
            bot,
            type=q("type") or None,
            q=q("q") or q("query") or None,
            offset=q_int("offset", 0),
            limit=q_int("limit", DEFAULT_MEMORY_LIMIT),
            corrections=q("filter").lower() == "corrections" or q("corrections").lower() in {"1", "true"},
        )
    if action == "retrieval":
        return build_retrieval(bot)
    if action == "history":
        return build_history(bot)
    if action == "events":
        hist = build_history(bot)
        return {"events": hist.get("events") or []}
    if action == "search":
        return build_search(bot, q("q") or q("query"), scope=q("scope") or "all", limit=q_int("limit", 40))
    if action == "snapshot":
        return build_snapshot(bot)
    return {"error": "not found"}


# --- internals ---


def _push_event(led: WorkingMemoryLedger, kind: str, **payload: Any) -> None:
    kind_u = str(kind or "").upper()
    if kind_u not in EVENT_TYPES:
        return
    ev = {"type": kind_u, "ts": utc_iso(), **payload}
    led.events.append(redact_secrets(ev))


def _push_event_unlocked_safe(led: WorkingMemoryLedger, kind: str, **payload: Any) -> None:
    with led.lock:
        _push_event(led, kind, **payload)


def _pos_int(value: Any) -> int | None:
    n = tel._as_int(value)
    if n is None or n <= 0:
        return None
    return int(n)


def _nonneg_int(value: Any) -> int | None:
    n = tel._as_int(value)
    if n is None or n < 0:
        return None
    return int(n)


def _title(text: str, n: int = 80) -> str:
    line = (text or "").strip().split("\n", 1)[0].strip()
    if len(line) <= n:
        return line
    return line[: n - 1].rstrip() + "…"


def _clip(text: str, n: int) -> str:
    s = text or ""
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _ts_iso(ts: int | float | None) -> str | None:
    if not ts:
        return None
    try:
        return utc_iso(float(ts))
    except (TypeError, ValueError, OSError):
        return None


def _parse_mem_id(raw: str) -> int | None:
    s = str(raw or "").strip()
    if s.startswith("mem_"):
        s = s[4:]
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _associated(tags: list[str] | None) -> list[str]:
    out = []
    for t in tags or []:
        low = str(t).lower()
        if low.startswith("skill:") or low in {"gaze", "wave", "look_at_user"}:
            out.append(str(t))
        elif low.startswith("skill"):
            out.append(str(t))
    return out


def _item(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("priority", "unknown")
    kwargs.setdefault("pinned", False)
    kwargs.setdefault("estimated", True)
    kwargs.setdefault("in_context", True)
    if "reason_codes" in kwargs:
        kwargs["why_loaded"] = why_loaded(kwargs.get("reason_codes"))
    return kwargs


def _sanitize_item(it: dict[str, Any]) -> dict[str, Any]:
    row = dict(it)
    src = str(row.get("source") or "")
    if src not in NAMED_SOURCES:
        row["source"] = "unknown"
    codes = []
    for c in row.get("reason_codes") or []:
        u = str(c or "").upper()
        if u in REASON_CODES:
            codes.append(u)
    row["reason_codes"] = codes
    row["why_loaded"] = why_loaded(codes)
    # Never ship full 262k blobs.
    if isinstance(row.get("content"), str):
        row["content"] = _clip(str(row["content"]), 500)
    for k in list(row.keys()):
        if str(k).lower().replace("-", "_") in _FORBIDDEN_KEYS:
            row.pop(k, None)
    return redact_secrets(row)


_SECTION_MARKERS = (
    ("[TEELA BODY NOW]", "body_state", "CURRENT_BODY_STATE", "body_state"),
    ("[TEELA WORLD NOW]", "world_state", "CURRENT_WORLD_STATE", "world_state"),
    ("[TEELA PERCEPTION EVENTS]", "perception", "PERCEPTION_RELEVANT", "perception"),
    ("[TEELA SAFETY NOW]", "safety_state", "SAFETY_RELEVANT", "MiniOS"),
    ("[SENSORIMOTOR EPISODE]", "sensorimotor", "PERCEPTION_RELEVANT", "episodic_memory"),
    ("[MOTOR SKILL]", "skills", "SKILL_MATCH", "skill"),
    ("[TEELA SPATIAL]", "world_state", "CURRENT_WORLD_STATE", "world_state"),
    ("How you look (BODY.md):", "core", "PINNED_CORE", "self_model"),
    ("Environment:", "workspace", "SYSTEM_REQUIRED", "MiniOS"),
    ("# Session memory", "active_task", "ACTIVE_TASK", "conversation_summary"),
    ("# Recalled facts", "retrieved_memory", "SEMANTIC_MATCH", "persistent_memory"),
    ("# Recent turns", "conversation", "RECENT_CONVERSATION", "conversation"),
    ("Likely relevant capability", "skills", "SKILL_MATCH", "skill"),
    ("Deliberation:", "core", "SYSTEM_REQUIRED", "system"),
)


def _items_from_payload(
    messages: list[Any],
    assembled: Any,
    now: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    items: list[dict[str, Any]] = []
    sections: dict[str, int] = {k: 0 for k in SECTION_KEYS}
    retrieved_by_text: dict[str, dict[str, Any]] = {}
    if assembled is not None:
        for it in getattr(assembled, "items", None) or []:
            if isinstance(it, dict) and it.get("title"):
                retrieved_by_text[str(it.get("title") or "").strip()[:80]] = it
        ret = getattr(assembled, "retrieval", None) or {}
        for row in list(ret.get("selected") or []):
            if isinstance(row, dict) and row.get("title"):
                retrieved_by_text[str(row.get("title") or "").strip()[:80]] = row

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").lower()
        text = _safe_message_text(msg)
        if not text:
            continue
        if role == "system":
            chunks = _split_system(text)
            for i, (sec, source, codes, pinned, chunk) in enumerate(chunks):
                n, est = count_section_tokens(chunk)
                iid = f"sys_{idx}_{i}_{sec}"
                title = {
                    "core": "Core identity / system instructions",
                    "body_state": "Current body state",
                    "workspace": "Workspace / MiniOS environment",
                    "active_task": "Session memory / active task",
                    "retrieved_memory": "Retrieved memories",
                    "conversation": "Recent conversation",
                    "skills": "Skill hint",
                }.get(sec, sec.replace("_", " ").title())
                it = _item(
                    id=iid,
                    section=sec,
                    type=_type_for_section(sec),
                    title=title,
                    token_count=n,
                    estimated=est,
                    source=source if source in NAMED_SOURCES else "unknown",
                    reason_codes=list(codes),
                    loaded_at=now,
                    pinned=bool(pinned),
                    last_used=now,
                    store=None,
                    record=None,
                )
                items.append(it)
                sections[sec] = sections.get(sec, 0) + n
        elif role in {"user", "assistant"}:
            n, est = count_section_tokens(text)
            items.append(
                _item(
                    id=f"turn_{idx}_{role}",
                    section="conversation",
                    type="conversation",
                    title=_title(text),
                    token_count=n,
                    estimated=est,
                    source=role if role in NAMED_SOURCES else "conversation",
                    reason_codes=["RECENT_CONVERSATION"],
                    loaded_at=now,
                    pinned=False,
                    last_used=now,
                )
            )
            sections["conversation"] = sections.get("conversation", 0) + n
        elif role in {"tool", "function"}:
            n, est = count_section_tokens(text)
            items.append(
                _item(
                    id=f"tool_{idx}",
                    section="tools",
                    type="tool_result",
                    title=_title(text) or "Tool result",
                    token_count=n,
                    estimated=est,
                    source="tool_result",
                    reason_codes=["SYSTEM_REQUIRED"],
                    loaded_at=now,
                    pinned=False,
                )
            )
            sections["tools"] = sections.get("tools", 0) + n
        else:
            n, est = count_section_tokens(text)
            items.append(
                _item(
                    id=f"other_{idx}",
                    section="unknown",
                    type="unknown",
                    title=_title(text) or "Unclassified",
                    token_count=n,
                    estimated=est,
                    source="unknown",
                    reason_codes=[],
                    loaded_at=now,
                    pinned=False,
                )
            )
            sections["unknown"] = sections.get("unknown", 0) + n
    return items, sections


def _split_system(text: str) -> list[tuple[str, str, list[str], bool, str]]:
    """Return list of (section, source, reason_codes, pinned, chunk)."""
    raw = text or ""
    hits: list[tuple[int, str, str, list[str], bool]] = []
    for marker, sec, code, source in _SECTION_MARKERS:
        p = raw.find(marker)
        if p >= 0:
            pinned = sec in {"core"} or code in {"PINNED_CORE", "ACTIVE_GOAL"}
            hits.append((p, sec, source, [code], pinned))
    hits.sort(key=lambda h: h[0])
    if not hits:
        return [("core", "system", ["PINNED_CORE", "SYSTEM_REQUIRED"], True, raw)]
    out: list[tuple[str, str, list[str], bool, str]] = []
    if hits[0][0] > 0:
        head = raw[: hits[0][0]].strip()
        if head:
            out.append(("core", "system", ["PINNED_CORE", "SYSTEM_REQUIRED"], True, head))
    for i, (pos, sec, source, codes, pinned) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(raw)
        chunk = raw[pos:end].strip()
        if not chunk:
            continue
        extra = list(codes)
        if sec == "core":
            extra = ["PINNED_CORE", "SYSTEM_REQUIRED"]
        if sec == "active_task":
            extra = ["ACTIVE_TASK", "TASK_CHECKPOINT"]
        if sec == "retrieved_memory":
            extra = ["SEMANTIC_MATCH"]
        out.append((sec, source, extra, pinned or sec == "core", chunk))
    return out


def _type_for_section(sec: str) -> str:
    return {
        "core": "self",
        "body_state": "body_state",
        "world_state": "world_state",
        "conversation": "conversation",
        "active_task": "task",
        "retrieved_memory": "unknown",
        "skills": "procedural",
        "perception": "perception",
        "safety_state": "safety",
        "sensorimotor": "sensorimotor_episode",
        "tools": "unknown",
        "workspace": "unknown",
    }.get(sec, "unknown")


def _safe_message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        bits: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = str(part.get("type") or "").lower()
            if ptype in _FORBIDDEN_KEYS or ptype in {"thought", "reasoning", "thinking"}:
                continue
            if ptype == "text" or "text" in part:
                bits.append(str(part.get("text") or ""))
        return "\n".join(b for b in bits if b)
    if content is None:
        return str(msg.get("text") or "")
    return str(content)


def _memory_counts(bot: Any) -> tuple[int | None, dict[str, int] | None, str | None]:
    mgr = getattr(bot, "memory", None)
    if mgr is None:
        return None, None, _UNAVAILABLE
    store = getattr(mgr, "store", None)
    if store is None:
        return None, None, _UNAVAILABLE
    try:
        types = store.type_counts()
        total = int(types.get("total") or store.count())
        return total, types, None
    except Exception:
        return None, None, _UNAVAILABLE


def _currently_loaded_memories(items: list[dict[str, Any]]) -> int:
    n = 0
    seen: set[str] = set()
    for it in items:
        if str(it.get("section") or "") != "retrieved_memory":
            continue
        iid = str(it.get("id") or "")
        if iid.startswith("sys_"):
            continue
        if iid and iid not in seen:
            seen.add(iid)
            n += 1
    return n


def _currently_loaded_memories_from_ids(loaded: set[str]) -> int:
    return len([i for i in loaded if str(i).startswith("mem_")])


def _body_snapshot(bot: Any) -> dict[str, Any] | None:
    getter = getattr(bot, "working_memory_body_snapshot", None)
    if callable(getter):
        try:
            snap = getter()
            return snap if isinstance(snap, dict) else None
        except Exception:
            return None
    try:
        from pathlib import Path

        import body_state as _bs

        bid = str(getattr(bot, "id", "") or "teela")
        root = getattr(bot, "root", None) or getattr(bot, "workspace", None) or "."
        store = _bs.store_for(bid, Path(str(root)) / "body_state.sqlite")
        snap = store.snapshot()
        return snap if isinstance(snap, dict) else None
    except Exception:
        return None


def _body_section(bot: Any) -> dict[str, Any]:
    try:
        import cognitive_profile as _cprof

        if not _cprof.has_cap(bot, "body_state"):
            return unavailable("Body state is not enabled for this agent profile")
    except Exception:
        pass
    try:
        snap = _body_snapshot(bot)
        if not isinstance(snap, dict) or not snap:
            return unavailable("Body state: Unavailable")
        age = tel._as_int(snap.get("freshness_ms"))
        stale = bool(age is not None and age > BODY_STALE_MS)
        confirmation = str(snap.get("confirmation") or snap.get("confirmation_status") or "")
        last_known = confirmation == "last_known"
        joints = snap.get("joints") if isinstance(snap.get("joints"), dict) else {}
        pan = _joint_deg(joints, "neck_pan")
        tilt = _joint_deg(joints, "neck_tilt")
        status = "stale" if stale else ("last_known" if last_known else "ok")
        out: dict[str, Any] = {
            "status": status,
            "pose": snap.get("pose"),
            "motion": snap.get("motion"),
            "waving": snap.get("waving"),
            "revision": snap.get("revision"),
            "freshness_ms": age,
            "updated_ago_ms": age,
            "confirmation": confirmation or None,
            "confirmation_status": confirmation or None,
            "authority": "MiniOS/body-state",
            "head": {
                "pan": pan if pan is not None else unavailable("Head pan not in body snapshot"),
                "tilt": tilt if tilt is not None else unavailable("Head tilt not in body snapshot"),
            },
            "gaze": unavailable("Gaze is not a first-class field in the current body snapshot"),
            "last_action": snap.get("last_action"),
            "mode": snap.get("mode"),
            "source": snap.get("source"),
        }
        if stale:
            out["reason"] = f"Stale — last update {(age or 0) / 1000.0:.1f}s ago"
        elif last_known:
            out["reason"] = "LAST_KNOWN_STATE — not confirmed by live telemetry since restart"
        return out
    except Exception as e:
        return unavailable(f"Body state: Unavailable ({e})")


def _embodied_sections(bot: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    world = unavailable("No world-state infrastructure in the current assembler")
    perception = unavailable("No perception-event log in the current assembler")
    safety = unavailable("Safety state is not enabled for this agent profile")
    sensorimotor = unavailable("Sensorimotor memory is not enabled for this agent profile")
    try:
        import cognitive_profile as _cprof
        import embodied_memory as emem
    except Exception:
        return world, perception, safety, sensorimotor
    if not _cprof.is_embodied(bot):
        world = unavailable("World state is not enabled for this agent profile")
        perception = unavailable("Perception is not enabled for this agent profile")
        return world, perception, safety, sensorimotor
    facade = emem.for_bot(bot)
    if _cprof.has_cap(bot, "world_state"):
        cur = facade.current_world()
        if cur:
            world = {
                "status": "ok",
                "revision": cur.get("revision"),
                "timestamp": cur.get("timestamp"),
                "source": "WORLD_STATE",
                "people": cur.get("people") or {},
                "objects": cur.get("objects") or {},
                "stale": cur.get("stale") or {},
                "hazards": cur.get("hazards") or [],
            }
        else:
            world = unavailable("No world-state infrastructure in the current assembler")
    if _cprof.has_cap(bot, "continuous_perception"):
        stats = facade.perception_stats()
        events = facade.events(limit=8)
        if stats is not None or events:
            perception = {
                "status": "ok",
                "event_count": (stats or {}).get("event_count", len(events)),
                "dropped_frames": (stats or {}).get("dropped_frames", 0),
                "events": events,
                "source": "PERCEPTION_EVENT",
            }
        else:
            perception = unavailable("No perception-event log in the current assembler")
    if _cprof.has_cap(bot, "safety_state"):
        snap = facade.safety_snapshot()
        if snap:
            safety = {"status": "ok", **snap}
        else:
            safety = unavailable("No safety-state runtime object")
    if _cprof.has_cap(bot, "sensorimotor_memory"):
        eps = facade.retrieve_episodes("", limit=5)
        if eps:
            sensorimotor = {
                "status": "ok",
                "stored": len(eps),
                "retrieved": [
                    {"episode_id": e.get("episode_id"), "goal": e.get("goal"), "success": (e.get("outcome") or {}).get("success")}
                    for e in eps
                ],
            }
        else:
            sensorimotor = unavailable("No sensorimotor episodes stored")
    return world, perception, safety, sensorimotor


def _joint_deg(joints: dict[str, Any], name: str) -> float | None:
    rec = joints.get(name)
    if not isinstance(rec, dict):
        return None
    try:
        return float(rec.get("actual"))
    except (TypeError, ValueError):
        return None


def _active_task_section(bot: Any, sections: dict[str, int]) -> dict[str, Any]:
    goal = None
    state = None
    step = None
    checkpoint = None
    rec = None
    mgr = getattr(bot, "memory", None)
    try:
        from pathlib import Path

        if mgr is not None:
            cp = Path(mgr.root) / "task_checkpoint.json"
            if cp.is_file():
                rec = json.loads(cp.read_text(encoding="utf-8"))
                if not isinstance(rec, dict):
                    rec = None
    except Exception:
        rec = None
    rec_status = str((rec or {}).get("status") or "").lower()
    if rec and rec_status not in {"done", "completed"} and rec.get("goal"):
        cost = sections.get("active_task") if sections else None
        return {
            "status": "ok",
            "goal": rec.get("goal"),
            "state": str(rec.get("status") or "RUNNING").upper(),
            "current_step": rec.get("current_step"),
            "checkpoint": rec.get("task_id") or rec.get("next_action"),
            "context_cost_tokens": cost,
            "context_cost_estimated": True if cost is not None else None,
        }
    if rec_status in {"done", "completed"}:
        # Checkpoint is authoritative: leftover physical_training momentum must
        # not keep a finished task visible in the manifest.
        return unavailable("No active task")
    ctx = None
    try:
        from teela_cl.deliberation import load_momentum

        root = None
        ws = getattr(bot, "workspace", None)
        if ws:
            from pathlib import Path

            root = Path(str(ws)) / ".teela"
        ctx = load_momentum(root)
        mom = str(getattr(ctx, "conversation_momentum", "") or "").strip().lower()
        active = mom in {"physical_training", "task", "working", "agent"} or bool(
            getattr(ctx, "user_feedback_expected", False)
        )
        if active and ctx.goal:
            goal = ctx.goal
            state = ctx.conversation_momentum or "RUNNING"
            step = ctx.active_context or None
        if ctx.last_attempt_id:
            checkpoint = ctx.last_attempt_id
    except Exception:
        pass
    if rec_status in {"done", "completed"} and not goal:
        return unavailable("No active task")
    if goal is None and mgr is not None:
        summary = getattr(mgr, "summary", None)
        if summary is not None and getattr(summary, "goal", None):
            goal = summary.goal
            state = state or "RUNNING"
    plan = getattr(bot, "plan", None)
    if isinstance(plan, dict) and plan.get("entries"):
        entries = plan.get("entries") or []
        current = next(
            (e for e in entries if str(e.get("status") or "") in {"in_progress", "inprogress"}),
            None,
        )
        if current and not step:
            step = current.get("content")
        if not goal:
            goal = (current or entries[0]).get("content")
            state = "RUNNING"
    if not goal:
        return unavailable("No active task")
    cost = sections.get("active_task") if sections else None
    return {
        "status": "ok",
        "goal": goal,
        "state": (state or "RUNNING").upper() if state else "RUNNING",
        "current_step": step,
        "checkpoint": checkpoint,
        "context_cost_tokens": cost,
        "context_cost_estimated": True if cost is not None else None,
    }


def _runtime_section(bot: Any, stats: dict[str, Any]) -> dict[str, Any]:
    tps = getattr(bot, "tps", None)
    try:
        tps_n = float(tps) if tps is not None else None
    except (TypeError, ValueError):
        tps_n = None
    if tps_n is not None and (tps_n != tps_n or tps_n < 0):  # noqa: PLR0124
        tps_n = None
    return {
        "model": str(getattr(bot, "model", "") or "") or None,
        "context": {
            "used": stats["used_tokens"],
            "capacity": stats["capacity_tokens"],
        },
        "prompt_processing_tok_s": unavailable("Runtime does not expose prompt-processing tok/s"),
        "generation_tok_s": tps_n if tps_n is not None else unavailable("No generation speed sample yet"),
        "vram": unavailable("Runtime does not expose VRAM"),
        "kv": unavailable("Runtime does not expose KV memory"),
        "session": "active" if getattr(bot, "status", None) else None,
        "speed_source": str(getattr(bot, "speed_source", "") or "") or None,
        "token_source": str(getattr(bot, "token_source", "") or "") or None,
        "context_source": str(getattr(bot, "context_source", "") or "") or None,
    }


def _corrections_section(bot: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
    loaded = []
    for it in items:
        codes = {str(c).upper() for c in (it.get("reason_codes") or [])}
        tags = {str(t).lower() for t in (it.get("tags") or [])}
        if "USER_CORRECTION" in codes or "correction" in tags:
            loaded.append(
                {
                    "id": it.get("id"),
                    "title": it.get("title"),
                    "currently_loaded": True,
                    "confidence": it.get("confidence") or it.get("relevance_score"),
                    "why_loaded": it.get("why_loaded"),
                }
            )
    stored = []
    try:
        from pathlib import Path
        from teela_cl.skill_store import SkillStore

        ws = getattr(bot, "workspace", None)
        if ws:
            store = SkillStore(Path(str(ws)) / ".teela", seed=False)
            for sk in store.all():
                if str(getattr(sk, "source", "") or "") in {"corrected", "correction", "user_correction"}:
                    stored.append(
                        {
                            "id": sk.skill_id,
                            "title": sk.name,
                            "currently_loaded": any(
                                str(it.get("id") or "") == sk.skill_id or sk.name in str(it.get("title") or "")
                                for it in items
                            ),
                            "confidence": sk.confidence,
                            "source": "user correction",
                            "learned": sk.created_at,
                        }
                    )
    except Exception:
        pass
    return {"loaded": loaded, "stored": stored, "loaded_count": len(loaded)}


def _skills_section(bot: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
    loaded = [it for it in items if str(it.get("section") or "") == "skills"]
    stored_n = None
    try:
        from pathlib import Path
        from teela_cl.skill_store import SkillStore

        ws = getattr(bot, "workspace", None)
        if ws:
            store = SkillStore(Path(str(ws)) / ".teela", seed=False)
            stored_n = len(store.all())
    except Exception:
        stored_n = None
    return {
        "loaded_count": len(loaded),
        "stored_total": stored_n,
        "loaded": [{"id": it.get("id"), "title": it.get("title")} for it in loaded],
    }


def _skills_search(bot: Any, needle: str, *, corrections: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    q = needle.lower()
    led = ledger_for(str(getattr(bot, "id", "") or ""))
    with led.lock:
        loaded = set(led.loaded_ids)
        titles = {str(it.get("title") or "").lower() for it in led.items}
    try:
        from pathlib import Path
        from teela_cl.skill_store import SkillStore

        ws = getattr(bot, "workspace", None)
        if not ws:
            return out
        store = SkillStore(Path(str(ws)) / ".teela", seed=False)
        for sk in store.all():
            if corrections and str(sk.source or "") not in {"corrected", "correction", "user_correction"}:
                continue
            blob = " ".join([sk.skill_id, sk.name, sk.semantic_description or ""]).lower()
            if q not in blob:
                continue
            in_ctx = sk.skill_id in loaded or sk.name.lower() in titles
            out.append(
                {
                    "id": sk.skill_id,
                    "title": sk.name,
                    "type": "procedural",
                    "presence": "IN CONTEXT" if in_ctx else "STORED ONLY",
                    "in_context": in_ctx,
                    "source": "skill",
                }
            )
    except Exception:
        return out
    return out


def _capture_body_event(bot: Any, led: WorkingMemoryLedger) -> None:
    body = _body_section(bot)
    if is_unavailable(body):
        return
    rev = tel._as_int(body.get("revision"))
    with led.lock:
        if rev is not None and rev != led.last_body_revision:
            _push_event(led, "BODY_STATE_UPDATED", revision=rev, pose=body.get("pose"), motion=body.get("motion"))
            led.last_body_revision = rev


def _capture_task_event(bot: Any, led: WorkingMemoryLedger) -> None:
    task = _active_task_section(bot, {})
    if is_unavailable(task):
        return
    # Once per distinct loaded checkpoint, not once per assemble.
    key = "|".join(
        str(task.get(k) or "")
        for k in ("goal", "state", "current_step", "checkpoint")
    )
    with led.lock:
        prev = getattr(led, "_task_key", None)
        if key and key != prev:
            _push_event(
                led,
                "TASK_CHECKPOINT_LOADED",
                goal=task.get("goal"),
                state=task.get("state"),
                checkpoint=task.get("checkpoint"),
            )
            led._task_key = key  # type: ignore[attr-defined]


def _redact(value: Any, ancestors: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            low = key.lower().replace("-", "_")
            if low in _FORBIDDEN_KEYS:
                continue
            if low in _SENSITIVE_KEYS or any(s in low for s in ("api_key", "password", "secret", "authorization")):
                out[key] = _REDACTED
                continue
            out[key] = _redact(v, ancestors + (low,))
        return out
    if isinstance(value, list):
        return [_redact(v, ancestors) for v in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact_text(text: str) -> str:
    s = text or ""
    s = _BEARER.sub("Bearer " + _REDACTED, s)
    s = _SECRET_IN_TEXT.sub(lambda m: f"{m.group(1)}: {_REDACTED}", s)
    return s
