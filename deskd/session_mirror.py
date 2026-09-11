"""Tail a Grok session updates.jsonl and emit only user text + final assistant text."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

EmitFn = Callable[[dict[str, Any]], None]


class SessionMirror:
    def __init__(
        self,
        sessions_root: Path,
        on_turn: Callable[[str, str], None],
        on_usage: Callable[[dict], None] | None = None,
    ) -> None:
        self.sessions_root = sessions_root
        self.on_turn = on_turn
        self.on_usage = on_usage
        self._stop = threading.Event()
        self._offset: dict[str, int] = {}
        self._user = ""
        self._asst = ""
        self._user_sent = False
        self._last_file: Path | None = None

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _latest(self) -> Path | None:
        if not self.sessions_root.is_dir():
            return None
        files = list(self.sessions_root.glob("**/updates.jsonl"))
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                path = self._latest()
                if path:
                    if self._last_file and path != self._last_file:
                        self._flush()
                    self._last_file = path
                    self._ingest(path)
            except Exception:
                pass
            self._stop.wait(0.35)

    def _ingest(self, path: Path) -> None:
        key = str(path)
        pos = self._offset.get(key, 0)
        with path.open("rb") as f:
            f.seek(pos)
            blob = f.read()
            self._offset[key] = f.tell()
        if not blob:
            return
        for raw in blob.splitlines():
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            self._handle(rec)

    def _handle(self, rec: dict[str, Any]) -> None:
        params = rec.get("params") or {}
        update = params.get("update") or {}
        kind = update.get("sessionUpdate") or ""
        content = update.get("content") or {}
        text = ""
        if isinstance(content, dict):
            text = content.get("text") or ""
        if kind == "user_message_chunk" and text:
            self._user += text
        elif kind == "agent_message_chunk" and text:
            if self._user and not self._user_sent:
                self.on_turn("user", self._user.strip())
                self._user_sent = True
            self._asst += text
        elif kind in ("agent_thought_chunk", "tool_call", "tool_call_update", "plan", "available_commands_update"):
            return
        elif kind == "turn_completed":
            if self.on_usage:
                try:
                    self.on_usage(update.get("usage") or params.get("_meta") or {})
                except Exception:
                    pass
            self._flush()

    def _flush(self) -> None:
        if self._user and not self._user_sent:
            self.on_turn("user", self._user.strip())
        if self._asst.strip():
            self.on_turn("assistant", self._asst.strip())
        self._user = ""
        self._asst = ""
        self._user_sent = False
