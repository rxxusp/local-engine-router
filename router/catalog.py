"""Runtime model catalog for local-engine-router.

The catalog is a durable, inspectable merge of static configuration, aliases,
engine hints, live engine model lists, parsed launch commands, and persisted
last-seen entries. It is intentionally owned by EngineManager so engine
lifecycle stays in engines.py while model discovery logic lives here.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import time
from dataclasses import dataclass, field
from typing import Any

from .config import RouterConfig, build_model_index


SOURCE_PRIORITY: dict[str, int] = {
    "static": 100,
    "alias": 90,
    "served_models": 80,
    "live": 70,
    "start_cmd": 60,
    "persisted": 10,
}


@dataclass
class CatalogEntry:
    id: str
    engine: str
    source: str
    first_seen: int
    last_seen: int
    last_live_status: str = "unknown"
    resolved_model: str | None = None
    stale: bool = False
    collisions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "engine": self.engine,
            "source": self.source,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "last_live_status": self.last_live_status,
            "stale": self.stale,
            "collisions": list(self.collisions),
        }
        if self.resolved_model:
            out["resolved_model"] = self.resolved_model
        return out


@dataclass
class ResolveResult:
    requested_model: str
    resolved_model: str
    engine: str | None
    source: str | None
    would_swap: bool
    reasons: list[str]
    collisions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "engine": self.engine,
            "source": self.source,
            "would_swap": self.would_swap,
            "reasons": list(self.reasons),
            "collisions": list(self.collisions),
        }


def _now() -> int:
    return int(time.time())


def _is_python_module(val: str) -> bool:
    if not val or "/" in val or val.lower().endswith(".gguf"):
        return False
    return "." in val


def _add_model_id(ids: set[str], val: str) -> None:
    if not val:
        return
    ids.add(val)
    if val.lower().endswith(".gguf"):
        base = os.path.basename(val)[: -len(".gguf")]
        if base:
            ids.add(base)


def _served_ids_from_start_cmd(start_cmd: list[str] | str) -> set[str]:
    if isinstance(start_cmd, str):
        try:
            argv = shlex.split(start_cmd)
        except ValueError:
            argv = start_cmd.split()
    else:
        argv = list(start_cmd or [])

    ids: set[str] = set()
    i = 0
    while i < len(argv):
        tok = argv[i]
        flag, eq, eq_val = tok.partition("=")
        if flag == "--served-model-name":
            if eq:
                _add_model_id(ids, eq_val)
                i += 1
            else:
                i += 1
                while i < len(argv) and not argv[i].startswith("-"):
                    _add_model_id(ids, argv[i])
                    i += 1
            continue
        if flag in ("-m", "--model", "--model-path"):
            if eq:
                val = eq_val
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                val = argv[i + 1]
                i += 2
            else:
                i += 1
                continue
            if flag == "-m" and _is_python_module(val):
                continue
            _add_model_id(ids, val)
            continue
        i += 1
    return ids


class ModelCatalog:
    def __init__(self, cfg: RouterConfig, engines: dict[str, Any]) -> None:
        self.cfg = cfg
        self.engines = engines
        self.entries: dict[str, CatalogEntry] = {}
        self._persisted: dict[str, CatalogEntry] = {}
        self._load_state()
        self.rebuild()

    def _engine_order(self) -> list[str]:
        if self.cfg.engines:
            return [s.key for s in self.cfg.engines if s.enabled and s.key in self.engines]
        return [k for k in self.cfg.engine_keys() if k in self.engines]

    def _load_state(self) -> None:
        try:
            with open(self.cfg.state_file) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return

        now = _now()
        ttl = float(getattr(self.cfg.discover, "state_ttl_s", 0.0) or 0.0)
        raw_catalog = data.get("catalog") if isinstance(data, dict) else None
        if isinstance(raw_catalog, dict):
            for model_id, body in raw_catalog.items():
                if not isinstance(body, dict):
                    continue
                engine = body.get("engine")
                if not isinstance(model_id, str) or not model_id or not engine:
                    continue
                last_seen = int(body.get("last_seen") or now)
                stale = bool(ttl and now - last_seen > ttl)
                if stale:
                    continue
                entry = CatalogEntry(
                    id=model_id,
                    engine=str(engine),
                    source=str(body.get("source") or "persisted"),
                    first_seen=int(body.get("first_seen") or last_seen),
                    last_seen=last_seen,
                    last_live_status=str(body.get("last_live_status") or "stale"),
                    resolved_model=body.get("resolved_model"),
                    stale=body.get("last_live_status") != "live",
                    collisions=list(body.get("collisions") or []),
                )
                self._persisted[model_id] = entry

        # Legacy compatibility: earlier discovery builds stored only
        # {engine: [model ids]} under seen_models.
        raw_seen = data.get("seen_models") if isinstance(data, dict) else None
        if isinstance(raw_seen, dict):
            for engine, ids in raw_seen.items():
                if engine not in self.engines or not isinstance(ids, list):
                    continue
                for model_id in ids:
                    if not model_id or not isinstance(model_id, str):
                        continue
                    self._persisted.setdefault(
                        model_id,
                        CatalogEntry(
                            id=model_id,
                            engine=str(engine),
                            source="persisted",
                            first_seen=now,
                            last_seen=now,
                            last_live_status="stale",
                            stale=True,
                        ),
                    )

    def seen_models(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {k: set() for k in self.engines}
        for entry in self._persisted.values():
            if entry.engine in out:
                out[entry.engine].add(entry.id)
        for entry in self.entries.values():
            if entry.engine in out and entry.source in {"live", "persisted"}:
                out[entry.engine].add(entry.id)
        return out

    def set_seen_models(self, engine_key: str, ids: set[str]) -> None:
        now = _now()
        for model_id in ids:
            old = self._persisted.get(model_id)
            self._persisted[model_id] = CatalogEntry(
                id=model_id,
                engine=engine_key,
                source="persisted",
                first_seen=old.first_seen if old else now,
                last_seen=now,
                last_live_status="live",
                stale=False,
                collisions=list(old.collisions) if old else [],
            )
        self.rebuild()

    def rebuild(self, live_by_engine: dict[str, set[str]] | None = None) -> dict[str, Any]:
        now = _now()
        candidates: list[CatalogEntry] = []

        for spec in self.cfg.models:
            candidates.append(
                CatalogEntry(
                    id=spec.id,
                    engine=spec.engine,
                    source="static",
                    first_seen=now,
                    last_seen=now,
                    last_live_status="configured",
                )
            )

        if self.cfg.discover.enabled:
            for key in self._engine_order():
                engine = self.engines.get(key)
                ecfg = getattr(engine, "cfg", None)
                if ecfg is None:
                    continue
                if getattr(ecfg, "discover_models", False):
                    for mid in getattr(ecfg, "served_models", None) or []:
                        candidates.append(self._candidate(mid, key, "served_models", now))
                    for mid in _served_ids_from_start_cmd(getattr(ecfg, "start_cmd", None) or []):
                        candidates.append(self._candidate(mid, key, "start_cmd", now))

            for key, ids in (live_by_engine or {}).items():
                for mid in ids:
                    candidates.append(self._candidate(mid, key, "live", now, "live"))

            for entry in self._persisted.values():
                if entry.engine in self.engines:
                    clone = CatalogEntry(**entry.to_dict())
                    clone.source = "persisted"
                    clone.stale = True
                    if clone.last_live_status == "live":
                        clone.last_live_status = "stale"
                    candidates.append(clone)

        merged: dict[str, CatalogEntry] = {}
        for cand in candidates:
            if not cand.id or cand.engine not in self.engines:
                continue
            current = merged.get(cand.id)
            if current is None:
                merged[cand.id] = cand
                continue
            winner = self._winner(current, cand)
            loser = cand if winner is current else current
            note = (
                f"{cand.id!r} also claimed by {loser.engine!r} via {loser.source}; "
                f"{winner.engine!r} wins ({self.cfg.discover.collision})"
            )
            if note not in winner.collisions:
                winner.collisions.append(note)
            merged[cand.id] = winner

        # Add aliases after target entries are known. An alias is a catalog entry
        # that resolves to its configured real id and inherits the target engine.
        for alias, target in (self.cfg.aliases or {}).items():
            target_entry = merged.get(target)
            if target_entry is None:
                continue
            collisions: list[str] = []
            if alias in merged:
                loser = merged[alias]
                collisions.append(
                    f"{alias!r} also claimed by {loser.engine!r} via {loser.source}; "
                    f"alias to {target!r} wins"
                )
            merged[alias] = CatalogEntry(
                id=alias,
                engine=target_entry.engine,
                source="alias",
                first_seen=now,
                last_seen=now,
                last_live_status=target_entry.last_live_status,
                resolved_model=target,
                stale=target_entry.stale,
                collisions=collisions,
            )

        self.entries = merged
        return self.summary()

    def _candidate(
        self,
        model_id: str,
        engine: str,
        source: str,
        now: int,
        live_status: str = "configured",
    ) -> CatalogEntry:
        old = self._persisted.get(model_id)
        return CatalogEntry(
            id=model_id,
            engine=engine,
            source=source,
            first_seen=old.first_seen if old and old.engine == engine else now,
            last_seen=now,
            last_live_status=live_status,
            stale=source == "persisted",
        )

    def _winner(self, a: CatalogEntry, b: CatalogEntry) -> CatalogEntry:
        pa = SOURCE_PRIORITY.get(a.source, 0)
        pb = SOURCE_PRIORITY.get(b.source, 0)
        if pa != pb:
            return a if pa > pb else b
        order = {key: i for i, key in enumerate(self._engine_order())}
        return a if order.get(a.engine, 10_000) <= order.get(b.engine, 10_000) else b

    async def refresh(self) -> dict[str, Any]:
        live: dict[str, set[str]] = {}
        if self.cfg.discover.enabled:
            async def probe(key, engine):
                try:
                    return key, set(await engine.available_models())
                except Exception:
                    return key, set()

            # Independent HTTP probes run together; merge in declaration order
            # so completion timing cannot affect collision resolution.
            results = await asyncio.gather(*(
                probe(key, engine) for key, engine in self.engines.items()
            ))
            for key, ids in results:
                live[key] = ids
                if ids:
                    for model_id in ids:
                        old = self._persisted.get(model_id)
                        now = _now()
                        self._persisted[model_id] = CatalogEntry(
                            id=model_id,
                            engine=key,
                            source="persisted",
                            first_seen=old.first_seen if old else now,
                            last_seen=now,
                            last_live_status="live",
                            stale=False,
                            collisions=list(old.collisions) if old else [],
                        )
        return self.rebuild(live)

    def owner_for(self, model_id: str) -> CatalogEntry | None:
        return self.entries.get(model_id)

    def resolve(
        self, model_id: str, *, active_engine: str | None = None
    ) -> ResolveResult:
        requested = model_id
        real = self.cfg.aliases.get(model_id, model_id)
        reasons: list[str] = []
        if real != requested:
            reasons.append(f"alias {requested!r} resolves to {real!r}")
        entry = self.entries.get(requested) or self.entries.get(real)
        if entry is None:
            return ResolveResult(
                requested_model=requested,
                resolved_model=real,
                engine=None,
                source=None,
                would_swap=False,
                reasons=reasons + ["model is not in the catalog"],
            )
        if entry.source == "alias" and entry.resolved_model:
            real = entry.resolved_model
        if entry.source == "persisted":
            reasons.append("using persisted stale catalog entry")
        if entry.collisions:
            reasons.append("catalog collision resolved by policy")
        return ResolveResult(
            requested_model=requested,
            resolved_model=real,
            engine=entry.engine,
            source=entry.source,
            would_swap=active_engine != entry.engine,
            reasons=reasons or [f"matched catalog source {entry.source!r}"],
            collisions=list(entry.collisions),
        )

    def summary(self) -> dict[str, Any]:
        entries = [e.to_dict() for e in sorted(self.entries.values(), key=lambda x: x.id)]
        engines: dict[str, list[str]] = {}
        for entry in self.entries.values():
            engines.setdefault(entry.engine, []).append(entry.id)
        return {
            "enabled": bool(self.cfg.discover.enabled),
            "collision": self.cfg.discover.collision,
            "models": entries,
            "engines": {k: sorted(v) for k, v in sorted(engines.items())},
        }

    def state_payload(self) -> dict[str, Any]:
        return {
            model_id: entry.to_dict()
            for model_id, entry in sorted(self.entries.items())
            if entry.source not in {"static", "alias"}
        }

    def openai_models(self, created: int) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []
        static = build_model_index(self.cfg)
        for entry in sorted(self.entries.values(), key=lambda e: e.id):
            item: dict[str, Any] = {
                "id": entry.id,
                "object": "model",
                "created": created,
                "owned_by": entry.engine,
                "name": entry.id,
            }
            spec = static.get(entry.id)
            if spec is not None:
                item["name"] = spec.display_name
                item["context_length"] = spec.context_length
            if entry.resolved_model:
                item["resolved_model"] = entry.resolved_model
            data.append(item)
        return data
