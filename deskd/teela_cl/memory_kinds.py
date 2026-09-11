"""Typed memories: knowledge, strategy, experience — not mixed with RSI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .records import ExperienceRecord, KnowledgeRecord, StrategyRecord, _now


class TypedMemory:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else None
        self.knowledge: list[KnowledgeRecord] = []
        self.strategies: list[StrategyRecord] = []
        self.experience: list[ExperienceRecord] = []
        self._load()

    def _paths(self) -> dict[str, Path] | None:
        if self.root is None:
            return None
        return {
            "knowledge": self.root / "knowledge.json",
            "strategy": self.root / "strategy.json",
            "experience": self.root / "experience.json",
        }

    def _load(self) -> None:
        paths = self._paths()
        if not paths:
            return
        for kind, path in paths.items():
            if not path.is_file():
                continue
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if kind == "knowledge":
                self.knowledge = [KnowledgeRecord(**r) for r in rows if isinstance(r, dict)]
            elif kind == "strategy":
                self.strategies = [StrategyRecord(**r) for r in rows if isinstance(r, dict)]
            else:
                self.experience = [ExperienceRecord(**{k: v for k, v in r.items() if k in ExperienceRecord.__dataclass_fields__}) for r in rows if isinstance(r, dict)]

    def save(self) -> None:
        paths = self._paths()
        if not paths:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict

        paths["knowledge"].write_text(json.dumps([asdict(x) for x in self.knowledge], indent=2), encoding="utf-8")
        paths["strategy"].write_text(json.dumps([asdict(x) for x in self.strategies], indent=2), encoding="utf-8")
        paths["experience"].write_text(json.dumps([asdict(x) for x in self.experience], indent=2), encoding="utf-8")

    def add_knowledge(self, text: str, tags: list[str] | None = None) -> KnowledgeRecord:
        rec = KnowledgeRecord(knowledge_id=f"k_{len(self.knowledge)+1}", text=text, tags=list(tags or []))
        self.knowledge.append(rec)
        self.save()
        return rec

    def add_strategy(self, description: str, applies_to: list[str] | None = None) -> StrategyRecord:
        rec = StrategyRecord(strategy_id=f"s_{len(self.strategies)+1}", description=description, applies_to=list(applies_to or []))
        self.strategies.append(rec)
        self.save()
        return rec

    def add_experience(self, rec: ExperienceRecord) -> ExperienceRecord:
        self.experience.append(rec)
        self.save()
        return rec

    def related(self, goal: str) -> list[ExperienceRecord]:
        g = (goal or "").lower()
        return [e for e in self.experience if g in e.goal.lower() or e.goal.lower() in g]
