"""Smart model picker: request classification + swap-aware model selection.

Active when ``routing_mode: smart`` (the default). For each eligible request
(a smart alias like ``smart``/``auto``/``default``, a well-known cloud model
name, or an unknown model id that would otherwise hit fallback routing) the
picker:

  1. classifies the request into weighted job dimensions using deterministic,
     low-latency signals only (endpoint, tools, response_format, code fences,
     stack traces, equations, prompt length, ...) — no GPU, no network;
  2. enumerates every locally-servable candidate model (static registry,
     discovery catalog, live engine tags);
  3. scores each candidate on capability fit (benchmark priors + local
     calibration + user metadata), context fit, engine residency, expected
     swap cost, and observed reliability;
  4. applies the swap-worth-it rule: the best non-resident candidate must beat
     the best already-resident candidate by ``smart.swap_margin`` — the router
     doesn't just ask which model is strongest, it asks whether that model is
     worth unloading the current engine, waiting for memory reclaim, and
     cold-starting another backend for THIS request.

The decision object carries the full per-candidate score breakdown, benchmark
provenance, the expected swap penalty, and a ranked fallback plan, so
``/admin/smart/resolve`` and ``routerctl explain smart`` can show exactly why
a model was picked.

This module deliberately does not import ``router.engines`` — the manager is
duck-typed — so the dependency graph stays acyclic (engines.py imports us).
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .benchmarks import BenchmarkStore
from .config import RouterConfig
from .model_identity import identify

log = logging.getLogger("router.smart")

# Job dimensions the classifier emits. "embedding" is a special shape marker
# (embeddings endpoints), not a quality axis; "speed" and "reliability" are
# runtime-preference dimensions rather than benchmark capabilities.
JOB_DIMENSIONS: tuple[str, ...] = (
    "general", "coding", "code_editing", "math", "reasoning", "tool_use",
    "writing", "summarization", "long_context", "json_structured",
    "speed", "reliability", "embedding",
)

# Job dimensions that map onto benchmark capability axes 1:1.
_CAPABILITY_DIMENSIONS: frozenset[str] = frozenset(
    {"general", "coding", "code_editing", "math", "reasoning", "tool_use",
     "writing", "summarization", "long_context", "json_structured"}
)

# Well-known cloud model-id shapes that clients send by default (SDK defaults,
# Claude Code, Cursor, ...). In smart mode these route through the picker so a
# stock client transparently reaches the best local model.
_CLOUD_MODEL_RES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^gpt-[345]",  # gpt-4o, gpt-4.1, gpt-5, gpt-3.5-turbo
        r"^chatgpt",
        r"^o[134](-mini|-pro|-preview)?($|-)",
        r"^claude",
        r"^gemini",
        r"^grok",
        r"^(text-)?davinci",
        r"^text-embedding-(ada|3)",
        r"^deepseek-(chat|reasoner)$",
        r"^(mistral|codestral)-(large|medium|small|tiny)(-latest|-\d{4})?$",
        r"^command-(a|r7b)",
        r"^(sonnet|opus|haiku)$",
        r"^(kimi|glm|qwen)-.*-(latest|preview)$",
    )
)

_DEFAULT_WEIGHTS: dict[str, float] = {
    "quality": 0.40, "speed": 0.10, "residency": 0.10,
    "swap_cost": 0.20, "reliability": 0.10, "context": 0.10,
}
_POLICY_WEIGHTS: dict[str, dict[str, float]] = {
    "balanced": _DEFAULT_WEIGHTS,
    "fast": {"quality": 0.25, "speed": 0.30, "residency": 0.15,
             "swap_cost": 0.20, "reliability": 0.05, "context": 0.05},
    "quality": {"quality": 0.60, "speed": 0.02, "residency": 0.03,
                "swap_cost": 0.15, "reliability": 0.10, "context": 0.10},
    "economy": {"quality": 0.20, "speed": 0.25, "residency": 0.20,
                "swap_cost": 0.25, "reliability": 0.05, "context": 0.05},
}

# How many characters of prompt text the classifier scans (front + back).
_SCAN_CHARS = 8000

_CODE_FENCE_RE = re.compile(r"```")
_STACK_TRACE_RE = re.compile(
    r"Traceback \(most recent call last\)|File \"[^\"]+\", line \d+"
    r"|^\s+at .+\(.+:\d+:\d+\)|panic:|segfault|NullPointerException",
    re.MULTILINE,
)
_DIFF_RE = re.compile(r"^(--- a/|\+\+\+ b/|@@ [-+,\d ]+ @@|diff --git)", re.MULTILINE)
_FILE_PATH_RE = re.compile(
    r"[\w./~-]+\.(?:py|js|jsx|ts|tsx|rs|go|java|kt|c|h|cc|cpp|hpp|rb|php|swift|"
    r"sh|bash|zsh|sql|yaml|yml|toml|json|css|html|vue|svelte|proto|tf)\b"
)
_CODE_KEYWORD_RE = re.compile(
    r"\b(def |function |import |from \w+ import|class \w+[({:]|#include\s*<|"
    r"SELECT .+ FROM|const \w+ =|let \w+ =|fn \w+\(|func \w+\(|package main|"
    r"pub fn|async def|=>\s*{)"
)
_EDIT_VERB_RE = re.compile(
    r"\b(refactor|fix (this|the|my) (bug|code|function|test|error)|"
    r"apply (this|the) (patch|diff|change)|rename|debug|edit (this|the|my))\b",
    re.IGNORECASE,
)
_MATH_RE = re.compile(
    r"\\frac|\\int|\\sum|\\sqrt|\$\$|\b(solve|equation|integral|derivative|"
    r"theorem|prove|calculate|probability)\b|\d+\s*[+*/^]\s*\d+|\d+\s*=\s*\d+",
    re.IGNORECASE,
)
_REASONING_RE = re.compile(
    r"\b(step[ -]by[ -]step|think (it )?through|logic puzzle|riddle|"
    r"explain why|reason about|deduce|chain of thought)\b",
    re.IGNORECASE,
)
_SUMMARIZE_RE = re.compile(
    r"\b(summari[sz]e|summary|tl;?dr|condense|key points|abstract for)\b",
    re.IGNORECASE,
)
_WRITING_RE = re.compile(
    r"\b(write|draft|compose) (a|an|the|my) "
    r"(story|poem|essay|blog|article|email|letter|post|novel|speech|bio)\b"
    r"|\brewrite\b|\bproofread\b|\bcopyedit\b",
    re.IGNORECASE,
)
_JSON_ASK_RE = re.compile(
    r"\b(json|schema)\b.{0,40}\b(output|format|respond|return|reply)\b"
    r"|\b(output|format|respond|return|reply)\b.{0,40}\bjson\b",
    re.IGNORECASE | re.DOTALL,
)


# --------------------------------------------------------------------------- #
# Request classification
# --------------------------------------------------------------------------- #
def _iter_text_parts(value: Any):
    """Yield the string fragments of a message content / prompt / input value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                yield item
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    yield text


def _has_image_parts(body: dict[str, Any]) -> bool:
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                    "image_url", "image", "input_image",
                ):
                    return True
        if isinstance(msg.get("images"), list) and msg["images"]:
            return True  # Ollama native image field
    return bool(body.get("images"))


def _gather_text(body: dict[str, Any]) -> tuple[str, int]:
    """Return (scan_text, estimated_prompt_tokens).

    ``scan_text`` is a bounded excerpt (head + tail) of all textual content;
    the token estimate covers the FULL text (chars/4)."""
    parts: list[str] = []
    total_chars = 0
    for key in ("system", "prompt", "input"):
        for text in _iter_text_parts(body.get(key)):
            parts.append(text)
            total_chars += len(text)
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        for text in _iter_text_parts(msg.get("content")):
            parts.append(text)
            total_chars += len(text)
    joined = "\n".join(parts)
    if len(joined) > _SCAN_CHARS:
        half = _SCAN_CHARS // 2
        scan = joined[:half] + "\n" + joined[-half:]
    else:
        scan = joined
    return scan, max(1, total_chars // 4)


def classify_request(path: str, body: dict[str, Any]) -> dict[str, float]:
    """Classify a request into normalized job-dimension weights.

    Deterministic and cheap: string/regex scans over a bounded excerpt plus
    structural signals. Returns {dimension -> weight}, weights summing to 1.
    """
    p = path.rstrip("/")
    if p.endswith(("/embeddings", "/embed")):
        return {"embedding": 1.0}

    scores: dict[str, float] = {dim: 0.0 for dim in JOB_DIMENSIONS}
    scores["general"] = 1.0  # baseline so an empty chat is still classifiable

    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        scores["tool_use"] += 2.5
        scores["json_structured"] += 0.5
        scores["reliability"] += 0.5
    if body.get("tool_choice") not in (None, "none"):
        scores["tool_use"] += 1.0

    rf = body.get("response_format")
    if isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"):
        scores["json_structured"] += 2.0
        scores["reliability"] += 0.5
    fmt = body.get("format")  # Ollama native structured output
    if fmt == "json" or isinstance(fmt, dict):
        scores["json_structured"] += 2.0
        scores["reliability"] += 0.5

    text, est_tokens = _gather_text(body)
    if text:
        fences = len(_CODE_FENCE_RE.findall(text))
        if fences:
            scores["coding"] += min(2.0, 0.75 * fences)
        if _STACK_TRACE_RE.search(text):
            scores["coding"] += 1.2
            scores["reasoning"] += 0.3
        if _DIFF_RE.search(text):
            scores["code_editing"] += 2.0
            scores["coding"] += 0.5
        if _FILE_PATH_RE.search(text):
            scores["coding"] += 0.6
        if _CODE_KEYWORD_RE.search(text):
            scores["coding"] += 0.8
        if _EDIT_VERB_RE.search(text) and (fences or _CODE_KEYWORD_RE.search(text)):
            scores["code_editing"] += 1.2
        if _MATH_RE.search(text):
            scores["math"] += 1.4
            scores["reasoning"] += 0.4
        if _REASONING_RE.search(text):
            scores["reasoning"] += 1.2
        if _SUMMARIZE_RE.search(text):
            scores["summarization"] += 1.5
        if _WRITING_RE.search(text):
            scores["writing"] += 1.5
        if _JSON_ASK_RE.search(text):
            scores["json_structured"] += 1.0

    if est_tokens > 32_000:
        scores["long_context"] += 2.0
    elif est_tokens > 8_000:
        scores["long_context"] += 1.0

    messages = body.get("messages")
    if isinstance(messages, list) and len(messages) > 6:
        scores["reliability"] += 0.3

    budget = body.get("max_completion_tokens", body.get("max_tokens"))
    if isinstance(budget, (int, float)) and not isinstance(budget, bool):
        if 0 < budget <= 256:
            scores["speed"] += 1.0
    if body.get("stream"):
        scores["speed"] += 0.3

    total = sum(scores.values())
    return {dim: round(v / total, 4) for dim, v in scores.items() if v > 0}


def is_cloud_model(model_id: str) -> bool:
    """True when *model_id* looks like a well-known cloud/API model name."""
    return any(p.search(model_id) for p in _CLOUD_MODEL_RES)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# --------------------------------------------------------------------------- #
# Health tracking
# --------------------------------------------------------------------------- #
@dataclass
class ModelHealth:
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    cooldown_until: float = 0.0  # wall-clock epoch seconds
    last_error: str = ""
    last_failure_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "cooldown_until": round(self.cooldown_until, 1),
            "last_error": self.last_error[-300:],
            "last_failure_at": round(self.last_failure_at, 1),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ModelHealth:
        if not isinstance(data, dict):
            return cls()
        try:
            return cls(
                consecutive_failures=int(data.get("consecutive_failures", 0)),
                total_failures=int(data.get("total_failures", 0)),
                total_successes=int(data.get("total_successes", 0)),
                cooldown_until=float(data.get("cooldown_until", 0.0)),
                last_error=str(data.get("last_error", "")),
                last_failure_at=float(data.get("last_failure_at", 0.0)),
            )
        except (TypeError, ValueError):
            return cls()


# --------------------------------------------------------------------------- #
# Decision objects
# --------------------------------------------------------------------------- #
@dataclass
class ScoredCandidate:
    model: str
    engine: str
    total: float
    components: dict[str, float]
    quality_detail: dict[str, float] = field(default_factory=dict)
    swap_cost_s: float = 0.0
    resident: bool = False
    excluded: str = ""  # non-empty = excluded, with the reason

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model": self.model,
            "engine": self.engine,
            "total": round(self.total, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "swap_cost_s": round(self.swap_cost_s, 1),
            "resident": self.resident,
        }
        if self.excluded:
            out["excluded"] = self.excluded
        return out


@dataclass
class SmartDecision:
    requested_model: str
    model: str
    engine: str
    policy: str
    confidence: float
    job: dict[str, float]
    would_swap: bool
    swap_cost_s: float
    reasons: list[str]
    candidates: list[ScoredCandidate]
    fallbacks: list[str]
    benchmarks: list[dict[str, Any]]
    retry: dict[str, Any]

    @property
    def primary_job(self) -> str:
        return max(self.job, key=self.job.get) if self.job else "general"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "smart",
            "requested_model": self.requested_model,
            "model": self.model,
            "engine": self.engine,
            "policy": self.policy,
            "confidence": round(self.confidence, 3),
            "job": self.job,
            "primary_job": self.primary_job,
            "would_swap": self.would_swap,
            "swap_cost_s": round(self.swap_cost_s, 1),
            "reasons": list(self.reasons),
            "candidates": [c.to_dict() for c in self.candidates],
            "fallbacks": list(self.fallbacks),
            "benchmarks": self.benchmarks,
            "retry": self.retry,
        }


# --------------------------------------------------------------------------- #
# The picker
# --------------------------------------------------------------------------- #
class SmartRouter:
    """Owns smart-mode state: benchmark store, health, calibration, estimates.

    Constructed and persisted by EngineManager; consulted by the HTTP layer
    before engine acquisition. ``mode`` starts as ``cfg.routing_mode`` and can
    be flipped at runtime via POST /admin/smart/mode (routerctl smart/manual).
    """

    def __init__(self, cfg: RouterConfig, manager: Any) -> None:
        self.cfg = cfg
        self.scfg = cfg.smart
        self.manager = manager
        self.mode: str = cfg.routing_mode
        self.benchmarks = BenchmarkStore(
            allow_network=cfg.smart.benchmarks.allow_network,
            cache_ttl_s=cfg.smart.benchmarks.cache_ttl_s,
        )
        self.health: dict[str, ModelHealth] = {}
        self.calibration: dict[str, dict[str, Any]] = {}
        self.last_pick: dict[str, Any] | None = None
        # Observed swap-duration EMA per engine key (seconds).
        self._swap_seconds: dict[str, float] = {}
        self.dirty = False

    # -- persistence ------------------------------------------------------ #
    def load_state(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        health = data.get("health")
        if isinstance(health, dict):
            for model, body in health.items():
                self.health[str(model)] = ModelHealth.from_dict(body)
        calibration = data.get("calibration")
        if isinstance(calibration, dict):
            for model, body in calibration.items():
                if isinstance(body, dict):
                    self.calibration[str(model)] = body
        swaps = data.get("swap_seconds")
        if isinstance(swaps, dict):
            for key, val in swaps.items():
                try:
                    self._swap_seconds[str(key)] = float(val)
                except (TypeError, ValueError):
                    continue

    def state_payload(self) -> dict[str, Any]:
        return {
            "health": {m: h.to_dict() for m, h in sorted(self.health.items())},
            "calibration": dict(sorted(self.calibration.items())),
            "swap_seconds": {
                k: round(v, 1) for k, v in sorted(self._swap_seconds.items())
            },
        }

    # -- runtime feedback -------------------------------------------------- #
    def note_swap(self, engine_key: str, duration_s: float, ok: bool) -> None:
        """Fold an observed swap duration into the per-engine estimate."""
        if not ok:
            return
        prev = self._swap_seconds.get(engine_key)
        self._swap_seconds[engine_key] = (
            duration_s if prev is None else 0.6 * prev + 0.4 * duration_s
        )
        self.dirty = True

    def record_success(self, model: str) -> None:
        h = self.health.setdefault(model, ModelHealth())
        h.total_successes += 1
        if h.consecutive_failures or h.cooldown_until:
            h.consecutive_failures = 0
            h.cooldown_until = 0.0
        self.dirty = True

    def record_failure(self, model: str, error: str) -> None:
        h = self.health.setdefault(model, ModelHealth())
        h.consecutive_failures += 1
        h.total_failures += 1
        h.last_error = str(error)[:500]
        h.last_failure_at = time.time()
        if h.consecutive_failures >= max(1, self.scfg.retry.failure_threshold):
            h.cooldown_until = time.time() + self.scfg.retry.cooldown_s
            log.warning(
                "smart: model %s entered cooldown for %.0fs after %d consecutive "
                "failures (%s)",
                model, self.scfg.retry.cooldown_s, h.consecutive_failures,
                h.last_error[:120],
            )
        self.dirty = True

    def in_cooldown(self, model: str) -> bool:
        h = self.health.get(model)
        return bool(h and h.cooldown_until > time.time())

    # -- status ------------------------------------------------------------ #
    def status_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "policy": self.scfg.policy,
            "aliases": list(self.scfg.aliases),
            "last_pick": self.last_pick,
            "health": {
                m: h.to_dict()
                for m, h in sorted(self.health.items())
                if h.total_failures or h.cooldown_until > time.time()
            },
            "calibrated_models": sorted(self.calibration),
            "benchmark_cache": {
                "cached_models": self.benchmarks.summary()["cached_models"],
                "allow_network": self.benchmarks.allow_network,
            },
        }

    # -- eligibility -------------------------------------------------------- #
    async def _known_exact(self, model: str) -> bool:
        """Is *model* an exact locally-configured/installed id (or alias)?"""
        if model in self.manager.index or model in (self.cfg.aliases or {}):
            return True
        try:
            if self.manager._discovered_index().get(model):
                return True
        except Exception:  # noqa: BLE001 - defensive; discovery is best-effort
            pass
        entry = self.manager.catalog.owner_for(model)
        if entry is not None:
            return True
        for engine in self.manager.engines.values():
            tags_fn = getattr(engine, "available_tags", None)
            if tags_fn is None:
                continue
            try:
                if model in await tags_fn():
                    return True
            except Exception:  # noqa: BLE001 - engine may be down
                continue
        return False

    async def maybe_pick(
        self, model: str, path: str, body: dict[str, Any], *, dry_run: bool = False
    ) -> SmartDecision | None:
        """Run smart selection when the request is eligible; None = route as
        before (exact/manual). Never raises. ``dry_run`` skips the last-pick
        bookkeeping so /admin/smart/resolve stays side-effect-free."""
        if self.mode != "smart":
            return None
        try:
            reason: str | None = None
            if model in self.scfg.aliases:
                # An exact model id shadows a same-named smart alias by design.
                if not await self._known_exact(model):
                    reason = f"smart alias {model!r}"
            if reason is None:
                known = await self._known_exact(model)
                if known:
                    if not self.scfg.override_exact_model_ids:
                        return None
                    reason = (
                        f"exact model {model!r} overridden "
                        "(smart.override_exact_model_ids)"
                    )
                elif is_cloud_model(model):
                    if not self.scfg.catch_cloud_models:
                        return None
                    reason = f"cloud model alias {model!r}"
                elif self.scfg.catch_unknown_models:
                    reason = f"unknown model id {model!r}"
                else:
                    return None
            decision = await self._decide(model, path, body, entry_reason=reason)
            if decision is not None and not dry_run:
                self.last_pick = {
                    "requested_model": decision.requested_model,
                    "model": decision.model,
                    "engine": decision.engine,
                    "confidence": round(decision.confidence, 3),
                    "primary_job": decision.primary_job,
                    "at": int(time.time()),
                }
                self.dirty = True
            return decision
        except Exception as exc:  # noqa: BLE001 - selection must never 500 a request
            log.exception("smart selection failed for %r: %s", model, exc)
            return None

    # -- candidate enumeration ---------------------------------------------- #
    async def _enumerate_candidates(self) -> dict[str, str]:
        """{model_id -> engine_key} of everything locally servable."""
        out: dict[str, str] = {}
        # Live tags from API-swap engines (Ollama etc.) — lowest precedence.
        for key, engine in self.manager.engines.items():
            tags_fn = getattr(engine, "available_tags", None)
            if tags_fn is None:
                continue
            try:
                for tag in await tags_fn():
                    out.setdefault(tag, key)
            except Exception:  # noqa: BLE001 - engine may be down
                continue
        # Discovery catalog + index.
        try:
            for model_id, engine_key in self.manager._discovered_index().items():
                out.setdefault(model_id, engine_key)
        except Exception:  # noqa: BLE001
            pass
        for entry in getattr(self.manager.catalog, "entries", {}).values():
            if entry.source == "alias":
                continue
            out.setdefault(entry.id, entry.engine)
        # Static registry wins.
        for spec in self.cfg.models:
            if spec.engine in self.manager.engines:
                out[spec.id] = spec.engine
        # Never offer alias names as candidates.
        for alias in (self.cfg.aliases or {}):
            out.pop(alias, None)
        for alias in self.scfg.aliases:
            out.pop(alias, None)
        return out

    # -- scoring -------------------------------------------------------------- #
    def _weights(self) -> dict[str, float]:
        policy = self.scfg.policy
        base = dict(
            self.scfg.policies.get(policy) or _POLICY_WEIGHTS.get(policy) or
            _DEFAULT_WEIGHTS
        )
        for key, val in _DEFAULT_WEIGHTS.items():
            base.setdefault(key, val * 0.5)
        base.update(self.scfg.weights)
        return base

    def _quality_score(
        self, model: str, job: dict[str, float]
    ) -> tuple[float, dict[str, float], float]:
        """(score, per-capability detail, benchmark match confidence)."""
        spec = self.manager.index.get(model)
        strengths: dict[str, float] = dict(getattr(spec, "strengths", None) or {})
        tier = getattr(spec, "quality_tier", None)
        tier_score = 0.15 + 0.15 * tier if tier else None
        calib_scores: dict[str, float] = (
            self.calibration.get(model, {}).get("scores") or {}
        )
        priors = (
            self.benchmarks.priors_for(model)
            if self.scfg.benchmarks.enabled
            else {}
        )
        match_conf = (
            self.benchmarks.match_confidence(model)
            if self.scfg.benchmarks.enabled
            else 0.0
        )
        if strengths or tier:
            match_conf = max(match_conf, 0.9)  # explicit user metadata
        if calib_scores:
            match_conf = max(match_conf, 0.8)

        detail: dict[str, float] = {}
        num = 0.0
        den = 0.0
        for dim, weight in job.items():
            if dim not in _CAPABILITY_DIMENSIONS:
                continue
            if dim in strengths:
                cap_score = strengths[dim]
            elif dim in calib_scores:
                cap_score = _clamp01(float(calib_scores[dim]))
            else:
                prior = priors.get(dim)
                if prior is not None and tier_score is not None:
                    # Blend user tier with the prior, weighted by prior confidence.
                    cap_score = prior[1] * prior[0] + (1 - prior[1]) * tier_score
                elif prior is not None:
                    cap_score = prior[0]
                elif tier_score is not None:
                    cap_score = tier_score
                else:
                    cap_score = 0.45
            detail[dim] = cap_score
            num += weight * cap_score
            den += weight
        score = num / den if den else 0.45
        return score, detail, match_conf

    def _speed_score(self, model: str) -> float:
        spec = self.manager.index.get(model)
        tier = getattr(spec, "speed_tier", None)
        if tier:
            return (tier - 1) / 4.0
        tps = self.calibration.get(model, {}).get("tokens_per_s")
        if isinstance(tps, (int, float)) and tps > 0:
            return _clamp01(math.log10(max(tps, 1.0) / 2.0) / 2.0)
        size_b = identify(model).size_b
        if size_b is None:
            return 0.5
        return _clamp01(1.05 - 0.28 * math.log10(max(size_b, 0.05) * 2.0))

    def swap_cost_estimate(self, engine_key: str) -> float:
        """Expected seconds to make *engine_key* active from the current state.

        0 when it is already active. Otherwise: observed swap-duration EMA if
        we've swapped to it before, else a conservative estimate from the
        engine's configured timeouts, plus a drain estimate for in-flight
        requests on the engines that would be stopped."""
        if self.manager.active_engine == engine_key:
            return 0.0
        observed = self._swap_seconds.get(engine_key)
        if observed is not None:
            cost = observed
        else:
            engine = self.manager.engines.get(engine_key)
            ecfg = getattr(engine, "cfg", None)
            start_timeout = getattr(ecfg, "start_timeout_s", None)
            load_timeout = getattr(ecfg, "load_timeout_s", None)
            if start_timeout:
                cost = 0.4 * float(start_timeout)
            elif load_timeout:
                cost = 0.5 * float(load_timeout)
            else:
                cost = 15.0
            cost += float(self.cfg.swap_memory_settle_timeout_s) * 0.2
        inflight_elsewhere = sum(
            n for key, n in getattr(self.manager, "_inflight", {}).items()
            if key != engine_key
        )
        cost += min(10.0, 2.0 * inflight_elsewhere)
        return cost

    def _context_fit(
        self, model: str, needed_tokens: int
    ) -> tuple[float, str]:
        """(score, exclusion reason or ''). Exclusion only on authoritative
        (configured) context lengths; guessed lengths just dent the score."""
        spec = self.manager.index.get(model)
        if spec is not None:
            ctx = int(spec.context_length)
            if needed_tokens > ctx:
                return 0.0, (
                    f"context does not fit ({needed_tokens} tokens needed, "
                    f"{ctx} available)"
                )
        else:
            ctx = 32_768  # unknown: assume a modern default, penalize only
        if needed_tokens <= 0.7 * ctx:
            return 1.0, ""
        return _clamp01((ctx - needed_tokens) / (0.3 * ctx)), ""

    def _shape_allows(self, model: str, job: dict[str, float]) -> str:
        """'' when the model fits the request shape, else the exclusion reason."""
        spec = self.manager.index.get(model)
        caps = set(getattr(spec, "capabilities", None) or [])
        identity = identify(model)
        is_embedding_model = "embedding" in caps or identity.is_embedding
        if "embedding" in job:
            if not is_embedding_model:
                return "not an embedding model"
            return ""
        if is_embedding_model and "embedding" not in job:
            return "embedding-only model on a generation request"
        if job.get("_vision"):
            if "vision" not in caps and not identity.is_vision:
                return "request contains images; model is not vision-capable"
        return ""

    # -- the decision --------------------------------------------------------- #
    async def _decide(
        self,
        requested: str,
        path: str,
        body: dict[str, Any],
        *,
        entry_reason: str | None = None,
    ) -> SmartDecision | None:
        job = classify_request(path, body)
        if _has_image_parts(body):
            job["_vision"] = 1.0

        _, est_tokens = _gather_text(body)
        budget = body.get("max_completion_tokens", body.get("max_tokens"))
        if not isinstance(budget, (int, float)) or isinstance(budget, bool):
            budget = 1024
        needed_tokens = est_tokens + int(max(0, budget))

        candidates = await self._enumerate_candidates()
        if not candidates:
            log.info("smart: no candidates for %r; falling back to legacy routing",
                     requested)
            return None

        weights = self._weights()
        wsum = sum(weights.values()) or 1.0
        job_speed_boost = job.get("speed", 0.0)
        job_reliability_boost = job.get("reliability", 0.0)

        scored: list[ScoredCandidate] = []
        now = time.time()
        for model, engine_key in sorted(candidates.items()):
            spec = self.manager.index.get(model)
            if spec is not None and not getattr(spec, "smart_enabled", True):
                continue
            resident = self.manager.active_engine == engine_key

            shape_reason = self._shape_allows(model, job)
            if shape_reason:
                scored.append(ScoredCandidate(
                    model=model, engine=engine_key, total=0.0, components={},
                    resident=resident, excluded=shape_reason,
                ))
                continue

            ctx_score, ctx_reason = self._context_fit(model, needed_tokens)
            if ctx_reason:
                scored.append(ScoredCandidate(
                    model=model, engine=engine_key, total=0.0, components={},
                    resident=resident, excluded=ctx_reason,
                ))
                continue

            h = self.health.get(model)
            if h and h.cooldown_until > now:
                scored.append(ScoredCandidate(
                    model=model, engine=engine_key, total=0.0, components={},
                    resident=resident,
                    excluded=(
                        f"in failure cooldown for another "
                        f"{h.cooldown_until - now:.0f}s ({h.last_error[:80]})"
                    ),
                ))
                continue

            quality, detail, match_conf = self._quality_score(model, job)
            speed = self._speed_score(model)
            swap_cost_s = self.swap_cost_estimate(engine_key)
            swap_score = 1.0 - _clamp01(
                swap_cost_s / max(self.scfg.swap_cost_horizon_s, 1.0)
            )
            reliability = 1.0
            if h:
                reliability -= 0.8 * _clamp01(
                    h.consecutive_failures
                    / max(1, self.scfg.retry.failure_threshold)
                )

            components = {
                "quality": quality,
                "speed": speed,
                "residency": 1.0 if resident else 0.0,
                "swap_cost": swap_score,
                "reliability": reliability,
                "context": ctx_score,
            }
            # Job-level speed/reliability preferences scale their components.
            eff = dict(weights)
            eff["speed"] = eff.get("speed", 0.0) * (1.0 + 2.0 * job_speed_boost)
            eff["reliability"] = eff.get("reliability", 0.0) * (
                1.0 + 2.0 * job_reliability_boost
            )
            eff_sum = sum(eff.values()) or wsum
            total = sum(
                eff.get(k, 0.0) * v for k, v in components.items()
            ) / eff_sum

            scored.append(ScoredCandidate(
                model=model, engine=engine_key, total=total,
                components=components, quality_detail=detail,
                swap_cost_s=swap_cost_s, resident=resident,
            ))
            # Stash per-candidate match confidence for the confidence calc.
            scored[-1].components["_match_confidence"] = match_conf

        viable = [c for c in scored if not c.excluded]
        if not viable:
            # Everything excluded (e.g. all in cooldown): let the least-bad
            # cooldown candidate through rather than failing the request.
            cooled = [
                c for c in scored
                if c.excluded.startswith("in failure cooldown")
            ]
            if not cooled:
                log.info("smart: every candidate excluded for %r; legacy routing",
                         requested)
                return None
            for c in cooled:
                c.total = 0.05 + (0.05 if c.resident else 0.0)
            viable = cooled

        viable.sort(key=lambda c: (-c.total, c.model))
        best = viable[0]
        reasons: list[str] = []
        if entry_reason:
            reasons.append(f"smart selection triggered by {entry_reason}")

        # Swap-worth-it rule: a non-resident winner must clear swap_margin over
        # the best resident candidate.
        resident_best = next((c for c in viable if c.resident), None)
        if (
            resident_best is not None
            and best is not resident_best
            and not best.resident
        ):
            gain = best.total - resident_best.total
            if gain < self.scfg.swap_margin:
                reasons.append(
                    f"kept resident {resident_best.model!r}: "
                    f"{best.model!r} scores +{gain:.3f} but the swap margin is "
                    f"{self.scfg.swap_margin} (estimated swap cost "
                    f"{best.swap_cost_s:.0f}s not worth it)"
                )
                best = resident_best
            else:
                reasons.append(
                    f"swapping engines: {best.model!r} clears the swap margin "
                    f"(+{gain:.3f} >= {self.scfg.swap_margin}) despite an "
                    f"estimated {best.swap_cost_s:.0f}s swap cost"
                )

        # Confidence: separation from the runner-up x benchmark match quality.
        runner_up = next((c for c in viable if c is not best), None)
        gap = (best.total - runner_up.total) if runner_up else 0.3
        match_conf = best.components.pop("_match_confidence", 0.0)
        for c in viable:
            c.components.pop("_match_confidence", None)
        confidence = _clamp01(
            (0.35 + _clamp01(gap * 4.0) * 0.65) * (0.45 + 0.55 * match_conf)
        )

        if (
            confidence < self.scfg.min_confidence
            and resident_best is not None
            and best is not resident_best
        ):
            reasons.append(
                f"confidence {confidence:.2f} below smart.min_confidence "
                f"{self.scfg.min_confidence}; keeping resident "
                f"{resident_best.model!r}"
            )
            best = resident_best

        reasons.append(
            f"picked {best.model!r} on {best.engine!r} "
            f"(score {best.total:.3f}, primary job "
            f"{max(job, key=job.get) if job else 'general'})"
        )

        fallbacks = [
            c.model for c in viable
            if c is not best and c.engine in self.manager.engines
        ][: max(0, self.scfg.retry.max_fallbacks)]

        bench_records = (
            [
                r.to_dict()
                for r in self.benchmarks.records_for(best.model)
                if r.capability in job
            ]
            if self.scfg.benchmarks.enabled
            else []
        )

        job_out = {k: v for k, v in job.items() if not k.startswith("_")}
        return SmartDecision(
            requested_model=requested,
            model=best.model,
            engine=best.engine,
            policy=self.scfg.policy,
            confidence=confidence,
            job=job_out,
            would_swap=self.manager.active_engine != best.engine,
            swap_cost_s=best.swap_cost_s,
            reasons=reasons,
            candidates=scored,
            fallbacks=fallbacks,
            benchmarks=bench_records,
            retry={
                "same_model_reload": self.scfg.retry.same_model_reload,
                "fall_forward": self.scfg.retry.fall_forward,
                "fallbacks": fallbacks,
                "never_after_streamed_bytes": True,
            },
        )


# --------------------------------------------------------------------------- #
# Local smoke calibration
# --------------------------------------------------------------------------- #
# Tiny deterministic probes measuring what benchmarks can't know about YOUR
# quantized/tuned/served copy: JSON compliance, tool-call formatting, short
# math, code syntax, instruction following — plus latency and tokens/sec.
_CALIBRATION_PROBES: tuple[dict[str, Any], ...] = (
    {
        "name": "json_compliance",
        "capability": "json_structured",
        "prompt": (
            'Return exactly this JSON object and nothing else: {"ok": true}'
        ),
        "check": "json",
    },
    {
        "name": "tool_call_format",
        "capability": "tool_use",
        "prompt": (
            "You must call a tool named get_weather with argument city set to "
            'Paris. Reply ONLY with the JSON arguments object, e.g. '
            '{"city": "..."}.'
        ),
        "check": "json",
    },
    {
        "name": "short_math",
        "capability": "math",
        "prompt": "What is 17 * 23? Reply with just the number.",
        "check": "contains:391",
    },
    {
        "name": "code_syntax",
        "capability": "coding",
        "prompt": (
            "Write a Python function add(a, b) that returns a + b. "
            "Reply with only the code, no backticks."
        ),
        "check": "python",
    },
    {
        "name": "instruction_following",
        "capability": "general",
        "prompt": "Reply with exactly the single word: PONG",
        "check": "contains:PONG",
    },
)


def _check_probe(check: str, content: str) -> bool:
    content = (content or "").strip()
    if check == "json":
        # Accept a fenced or bare JSON object anywhere in the reply.
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return False
        try:
            json.loads(m.group(0))
            return True
        except ValueError:
            return False
    if check.startswith("contains:"):
        return check[len("contains:"):] in content
    if check == "python":
        code = re.sub(r"^```(?:python)?|```$", "", content, flags=re.MULTILINE)
        try:
            compile(code, "<calibration>", "exec")
            return True
        except SyntaxError:
            return False
    return False


async def run_calibration(
    smart: SmartRouter, client: Any, model_id: str
) -> dict[str, Any]:
    """Run the one-time smoke calibration for *model_id* through its engine.

    Acquires the engine (this can trigger a swap — calibration is an explicit
    admin action, never automatic), sends the tiny probe battery, and stores
    per-capability pass scores plus latency/tokens-per-second in the smart
    state. Returns the stored record."""
    manager = smart.manager
    engine = await manager.acquire(model_id)
    results: dict[str, Any] = {"probes": {}, "scores": {}}
    latencies: list[float] = []
    tokens_per_s: list[float] = []
    try:
        for probe in _CALIBRATION_PROBES:
            payload = {
                "model": manager.resolve_model_id(model_id),
                "messages": [{"role": "user", "content": probe["prompt"]}],
                "max_tokens": 80,
                "temperature": 0,
                "stream": False,
            }
            t0 = time.monotonic()
            passed = False
            error = ""
            try:
                resp = await client.post(
                    f"{engine.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=120.0,
                )
                dt = time.monotonic() - t0
                latencies.append(dt)
                data = resp.json()
                content = (
                    (data.get("choices") or [{}])[0]
                    .get("message", {})
                    .get("content", "")
                ) or ""
                passed = _check_probe(probe["check"], content)
                usage = data.get("usage") or {}
                completion_tokens = usage.get("completion_tokens")
                if isinstance(completion_tokens, int) and dt > 0:
                    tokens_per_s.append(completion_tokens / dt)
            except Exception as exc:  # noqa: BLE001 - a probe failure is data
                error = str(exc)[:200]
            results["probes"][probe["name"]] = {
                "passed": passed, **({"error": error} if error else {}),
            }
            cap = probe["capability"]
            prev = results["scores"].get(cap)
            score = 0.85 if passed else 0.25
            results["scores"][cap] = (
                score if prev is None else (prev + score) / 2.0
            )
    finally:
        await manager.release(engine.key)

    if latencies:
        results["latency_ms"] = round(1000 * sum(latencies) / len(latencies), 1)
    if tokens_per_s:
        results["tokens_per_s"] = round(sum(tokens_per_s) / len(tokens_per_s), 1)
    results["at"] = int(time.time())
    smart.calibration[model_id] = results
    smart.dirty = True
    # Persist through the manager so the calibration survives restarts.
    persist = getattr(manager, "_persist", None)
    if persist is not None:
        persist()
    return results
