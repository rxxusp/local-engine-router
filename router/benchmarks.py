"""Benchmark intelligence for the smart model picker.

Public benchmark results are used as *priors* on how good a local model is at
each job dimension — never as ground truth. The flow:

  1. A model id is canonicalized by :mod:`router.model_identity`.
  2. :class:`BenchmarkStore` is asked for capability priors for that identity.
  3. The store answers from its cache (persisted in the router ``state_file``
     across restarts) or, on a miss, asks each registered
     :class:`BenchmarkProvider` once and caches the result — one fetch per
     canonical model/version, ever, unless explicitly refreshed.

The only provider shipped in v1 is :class:`BuiltinPriorsProvider`: a curated,
offline table distilled from public leaderboards (Artificial Analysis,
LMArena / Open LLM Leaderboard, Aider Polyglot, LiveCodeBench, BigCodeBench,
SWE-bench, BFCL, TAU-bench, MathArena / AIME aggregates, IFEval, RULER). It
needs no network, which keeps fresh installs deterministic; providers that DO
fetch over the network can be registered via :func:`register_provider` and are
only consulted when ``smart.benchmarks.allow_network`` is true.

Obscure models are never excluded: a community fine-tune or abliterated
variant inherits its base model's scores at reduced confidence, a quantized
repack inherits the unquantized scores minus a quant penalty, and a completely
unrecognized id falls back to a flat low-confidence prior so local calibration
and observed reliability can still differentiate it.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from .model_identity import ModelIdentity, identify, quant_quality_penalty

log = logging.getLogger("router.benchmarks")

# The capability axes benchmark priors are expressed in. These deliberately
# mirror the job dimensions produced by the request classifier in smart.py
# (minus the runtime-only dimensions: speed / reliability, which come from
# observed behaviour, not benchmarks).
CAPABILITIES: tuple[str, ...] = (
    "general",
    "coding",
    "code_editing",
    "math",
    "reasoning",
    "tool_use",
    "writing",
    "summarization",
    "long_context",
    "json_structured",
)

# Provenance: which public benchmarks each capability prior is distilled from.
_CAPABILITY_SOURCES: dict[str, tuple[str, str]] = {
    "general": (
        "Artificial Analysis Intelligence Index; LMArena Elo",
        "https://artificialanalysis.ai/models",
    ),
    "coding": (
        "LiveCodeBench; BigCodeBench; SWE-bench Verified",
        "https://livecodebench.github.io/leaderboard.html",
    ),
    "code_editing": (
        "Aider Polyglot leaderboard",
        "https://aider.chat/docs/leaderboards/",
    ),
    "math": (
        "AIME / MATH-500 aggregates; MathArena",
        "https://matharena.ai/",
    ),
    "reasoning": (
        "GPQA Diamond; LiveBench reasoning",
        "https://livebench.ai/",
    ),
    "tool_use": (
        "Berkeley Function-Calling Leaderboard (BFCL); TAU-bench",
        "https://gorilla.cs.berkeley.edu/leaderboard.html",
    ),
    "writing": (
        "LMArena creative writing; IFEval; MultiChallenge",
        "https://lmarena.ai/leaderboard",
    ),
    "summarization": (
        "HELM summarization aggregates",
        "https://crfm.stanford.edu/helm/",
    ),
    "long_context": (
        "RULER; LongBench",
        "https://github.com/NVIDIA/RULER",
    ),
    "json_structured": (
        "IFEval strict; structured-output evals",
        "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard",
    ),
}

# Date stamp of the curated builtin snapshot (bump when the table is retuned).
_BUILTIN_SNAPSHOT = "2026-07-01"


@dataclass
class BenchmarkRecord:
    """One capability prior with provenance, as stored in the cache."""

    source: str            # provider name, e.g. "builtin-priors"
    benchmark: str         # public benchmark(s) the prior is distilled from
    capability: str        # one of CAPABILITIES
    score: float           # normalized 0..1
    confidence: float      # 0..1 match confidence (exact > inherited > guess)
    match: str             # "exact" | "interpolated" | "inherited" | "heuristic"
    canonical_model: str   # canonical id the record was matched against
    date: str = _BUILTIN_SNAPSHOT
    url: str = ""
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source": self.source,
            "benchmark": self.benchmark,
            "capability": self.capability,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "match": self.match,
            "canonical_model": self.canonical_model,
            "date": self.date,
            "url": self.url,
        }
        if self.rank is not None:
            out["rank"] = self.rank
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkRecord | None:
        try:
            return cls(
                source=str(data["source"]),
                benchmark=str(data.get("benchmark", "")),
                capability=str(data["capability"]),
                score=float(data["score"]),
                confidence=float(data.get("confidence", 0.5)),
                match=str(data.get("match", "exact")),
                canonical_model=str(data.get("canonical_model", "")),
                date=str(data.get("date", _BUILTIN_SNAPSHOT)),
                url=str(data.get("url", "")),
                rank=data.get("rank"),
            )
        except (KeyError, TypeError, ValueError):
            return None


class BenchmarkProvider(Protocol):
    """A source of benchmark records for a canonical model identity."""

    name: str
    requires_network: bool

    def fetch(self, identity: ModelIdentity) -> list[BenchmarkRecord]:
        """Return records for *identity* (possibly empty). Must not raise."""
        ...  # pragma: no cover - protocol


# --------------------------------------------------------------------------- #
# Builtin curated priors
# --------------------------------------------------------------------------- #

def _size_baseline(size_b: float) -> float:
    """Baseline general-capability score from parameter count alone.

    Calibrated against mid-2026 aggregates: a ~7B instruct model scores ~0.50,
    a ~70B ~0.69, a frontier-scale open model (~670B) ~0.87."""
    return max(0.05, min(0.92, 0.34 + 0.19 * math.log10(max(size_b, 0.05))))


# Family profiles: a per-family quality offset applied to the size baseline,
# per-capability offsets on top of that, and the parameter sizes the family
# actually ships in (used for exact-vs-interpolated match confidence).
# Distilled from the public leaderboards listed in _CAPABILITY_SOURCES.
_FAMILY_PROFILES: dict[str, dict[str, Any]] = {
    # family: {tier, caps: {capability: extra offset}, sizes: [...]}
    "qwen3":        {"tier": 0.06, "caps": {"coding": 0.03, "math": 0.05, "tool_use": 0.04, "reasoning": 0.04}, "sizes": [0.6, 1.7, 4, 8, 14, 32, 235]},
    "qwen2.5":      {"tier": 0.03, "caps": {"coding": 0.02, "math": 0.04, "json_structured": 0.03}, "sizes": [0.5, 1.5, 3, 7, 14, 32, 72]},
    "qwen2":        {"tier": 0.00, "caps": {}, "sizes": [0.5, 1.5, 7, 72]},
    "qwq":          {"tier": 0.03, "caps": {"math": 0.14, "reasoning": 0.13, "writing": -0.04}, "sizes": [32]},
    "qwen":         {"tier": -0.06, "caps": {}, "sizes": [1.8, 7, 14, 72]},
    "llama3.3":     {"tier": 0.05, "caps": {"tool_use": 0.04, "writing": 0.03}, "sizes": [70]},
    "llama3.2":     {"tier": 0.02, "caps": {}, "sizes": [1, 3, 11, 90]},
    "llama3.1":     {"tier": 0.02, "caps": {"tool_use": 0.04, "long_context": 0.04, "writing": 0.03}, "sizes": [8, 70, 405]},
    "llama3":       {"tier": 0.00, "caps": {"writing": 0.03}, "sizes": [8, 70]},
    "llama2":       {"tier": -0.12, "caps": {"long_context": -0.08}, "sizes": [7, 13, 70]},
    "codellama":    {"tier": -0.08, "caps": {"coding": 0.14, "code_editing": 0.10, "writing": -0.08}, "sizes": [7, 13, 34]},
    "tinyllama":    {"tier": -0.08, "caps": {}, "sizes": [1.1]},
    "deepseek-r1":  {"tier": 0.04, "caps": {"math": 0.13, "reasoning": 0.13, "coding": 0.05, "tool_use": -0.04}, "sizes": [1.5, 7, 8, 14, 32, 70, 671]},
    "deepseek-v3":  {"tier": 0.06, "caps": {"coding": 0.05, "math": 0.05, "tool_use": 0.03}, "sizes": [671]},
    "deepseek-v2":  {"tier": 0.00, "caps": {}, "sizes": [16, 236]},
    "deepseek-coder": {"tier": -0.02, "caps": {"coding": 0.14, "code_editing": 0.12, "writing": -0.08}, "sizes": [1.3, 6.7, 33]},
    "deepseek":     {"tier": -0.04, "caps": {}, "sizes": [7, 67]},
    "mixtral":      {"tier": -0.03, "caps": {}, "sizes": [56, 176]},
    "mistral-small": {"tier": 0.03, "caps": {"tool_use": 0.03}, "sizes": [22, 24]},
    "mistral-large": {"tier": 0.04, "caps": {"tool_use": 0.04}, "sizes": [123]},
    "mistral-nemo": {"tier": 0.01, "caps": {}, "sizes": [12]},
    "ministral":    {"tier": 0.01, "caps": {}, "sizes": [3, 8]},
    "magistral":    {"tier": 0.03, "caps": {"math": 0.10, "reasoning": 0.10}, "sizes": [24]},
    "devstral":     {"tier": 0.03, "caps": {"coding": 0.12, "code_editing": 0.13, "tool_use": 0.06}, "sizes": [24]},
    "codestral":    {"tier": 0.00, "caps": {"coding": 0.13, "code_editing": 0.11, "writing": -0.06}, "sizes": [22]},
    "mistral":      {"tier": -0.03, "caps": {}, "sizes": [7]},
    "gemma3n":      {"tier": 0.03, "caps": {}, "sizes": [2, 4]},
    "gemma3":       {"tier": 0.04, "caps": {"writing": 0.05, "long_context": 0.03}, "sizes": [1, 4, 12, 27]},
    "gemma2":       {"tier": 0.00, "caps": {"writing": 0.04}, "sizes": [2, 9, 27]},
    "gemma":        {"tier": -0.07, "caps": {}, "sizes": [2, 7]},
    "phi4":         {"tier": 0.05, "caps": {"math": 0.09, "reasoning": 0.06, "tool_use": -0.03}, "sizes": [3.8, 14]},
    "phi3.5":       {"tier": 0.02, "caps": {"math": 0.05}, "sizes": [3.8]},
    "phi3":         {"tier": 0.00, "caps": {"math": 0.04}, "sizes": [3.8, 14]},
    "phi2":         {"tier": -0.06, "caps": {}, "sizes": [2.7]},
    "gpt-oss":      {"tier": 0.06, "caps": {"reasoning": 0.08, "math": 0.07, "tool_use": 0.06}, "sizes": [20, 120]},
    "glm4":         {"tier": 0.03, "caps": {"coding": 0.05, "tool_use": 0.05}, "sizes": [9, 32]},
    "granite":      {"tier": 0.00, "caps": {"json_structured": 0.04, "summarization": 0.03}, "sizes": [2, 8]},
    "command-r-plus": {"tier": 0.01, "caps": {"tool_use": 0.05, "summarization": 0.05, "long_context": 0.04}, "sizes": [104]},
    "command-r":    {"tier": 0.00, "caps": {"tool_use": 0.05, "summarization": 0.05, "long_context": 0.04}, "sizes": [35]},
    "smollm3":      {"tier": 0.02, "caps": {}, "sizes": [3]},
    "smollm2":      {"tier": 0.00, "caps": {}, "sizes": [0.135, 0.36, 1.7]},
    "smollm":       {"tier": -0.04, "caps": {}, "sizes": [0.135, 0.36, 1.7]},
    "starcoder2":   {"tier": -0.04, "caps": {"coding": 0.13, "code_editing": 0.08, "writing": -0.10}, "sizes": [3, 7, 15]},
    "starcoder":    {"tier": -0.10, "caps": {"coding": 0.12, "writing": -0.10}, "sizes": [15]},
    "olmo2":        {"tier": 0.00, "caps": {}, "sizes": [7, 13, 32]},
    "olmo":         {"tier": -0.06, "caps": {}, "sizes": [7]},
    "internlm2":    {"tier": -0.01, "caps": {}, "sizes": [7, 20]},
    "internlm":     {"tier": -0.05, "caps": {}, "sizes": [7, 20]},
    "falcon3":      {"tier": 0.00, "caps": {}, "sizes": [1, 3, 7, 10]},
    "falcon":       {"tier": -0.10, "caps": {}, "sizes": [7, 40, 180]},
    "yi1.5":        {"tier": 0.00, "caps": {}, "sizes": [6, 9, 34]},
    "yi":           {"tier": -0.04, "caps": {}, "sizes": [6, 34]},
    "minicpm":      {"tier": 0.00, "caps": {}, "sizes": [2.4, 4]},
    "kimi":         {"tier": 0.05, "caps": {"coding": 0.05, "tool_use": 0.05}, "sizes": [1000]},
    "seed-oss":     {"tier": 0.04, "caps": {"reasoning": 0.05}, "sizes": [36]},
    "ernie":        {"tier": 0.02, "caps": {}, "sizes": [21, 300]},
    "exaone":       {"tier": 0.01, "caps": {}, "sizes": [2.4, 7.8, 32]},
    "aya":          {"tier": -0.01, "caps": {"writing": 0.04}, "sizes": [8, 32]},
    "solar":        {"tier": -0.04, "caps": {}, "sizes": [10.7]},
    "dbrx":         {"tier": -0.02, "caps": {}, "sizes": [132]},
    "hunyuan":      {"tier": 0.02, "caps": {}, "sizes": [7, 80]},
    "pixtral":      {"tier": 0.01, "caps": {}, "sizes": [12, 124]},
    "moondream":    {"tier": -0.06, "caps": {}, "sizes": [1.8]},
    "llava":        {"tier": -0.06, "caps": {}, "sizes": [7, 13, 34]},
}

# Variant adjustments applied on top of family+size (a coder variant of a
# generalist family gains coding, loses a bit of prose, etc.).
_VARIANT_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "code": {"coding": 0.12, "code_editing": 0.12, "writing": -0.06, "summarization": -0.03},
    "base": {"general": -0.10, "tool_use": -0.15, "json_structured": -0.12, "writing": -0.08},
    "vision": {},
    "instruct": {},
    "": {},
}

# Default long-context aptitude bonus for families with >=128k training windows
# is folded into the per-family caps above; nothing dynamic here.

_FLAT_UNKNOWN_SCORE = 0.45  # prior for a completely unrecognized model id
_FLAT_UNKNOWN_CONFIDENCE = 0.2


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class BuiltinPriorsProvider:
    """Offline curated priors distilled from public leaderboards."""

    name = "builtin-priors"
    requires_network = False

    def fetch(self, identity: ModelIdentity) -> list[BenchmarkRecord]:
        scores, confidence, match = self._scores_for(identity)
        records: list[BenchmarkRecord] = []
        for cap in CAPABILITIES:
            benchmark, url = _CAPABILITY_SOURCES[cap]
            records.append(
                BenchmarkRecord(
                    source=self.name,
                    benchmark=benchmark,
                    capability=cap,
                    score=_clamp01(scores.get(cap, _FLAT_UNKNOWN_SCORE)),
                    confidence=confidence,
                    match=match,
                    canonical_model=identity.canonical,
                    url=url,
                )
            )
        return records

    # -- scoring internals ------------------------------------------------ #
    def _scores_for(
        self, identity: ModelIdentity
    ) -> tuple[dict[str, float], float, str]:
        family = identity.family
        profile = _FAMILY_PROFILES.get(family) if family else None

        # Completely unrecognized: flat prior, let calibration differentiate.
        if profile is None and identity.size_b is None:
            return (
                {cap: _FLAT_UNKNOWN_SCORE for cap in CAPABILITIES},
                _FLAT_UNKNOWN_CONFIDENCE,
                "heuristic",
            )

        # Known size but unknown family: size-only baseline, low confidence.
        if profile is None:
            base = _size_baseline(identity.size_b or 7.0)
            return ({cap: base for cap in CAPABILITIES}, 0.3, "heuristic")

        sizes: list[float] = profile.get("sizes", [])
        size_b = identity.size_b
        if size_b is None:
            # Family known, size not stated: assume the family's most common
            # (median) shipped size.
            size_b = sorted(sizes)[len(sizes) // 2] if sizes else 7.0
            match = "interpolated"
            confidence = 0.5
        elif any(abs(size_b - s) / max(s, 0.05) < 0.15 for s in sizes):
            match = "exact"
            confidence = 0.75
        else:
            match = "interpolated"
            confidence = 0.6

        base = _size_baseline(size_b) + float(profile.get("tier", 0.0))
        caps: dict[str, float] = profile.get("caps", {})
        variant_adj = _VARIANT_ADJUSTMENTS.get(identity.variant, {})

        # NOTE: no quant penalty here — the cache key (canonical id) is
        # quant-free, so quant loss is applied per-spelling in priors_for().
        scores = {
            cap: _clamp01(base + caps.get(cap, 0.0) + variant_adj.get(cap, 0.0))
            for cap in CAPABILITIES
        }

        # Fine-tunes / abliterated variants inherit the base model's scores at
        # reduced confidence (the tune may help or hurt; we can't know without
        # local calibration). Abliteration additionally costs a small quality
        # haircut — refusal-ablation measurably dents benchmark scores.
        if identity.fine_tune:
            confidence *= 0.75
            match = "inherited"
        if identity.abliterated:
            confidence *= 0.8
            scores = {cap: _clamp01(s - 0.03) for cap, s in scores.items()}
            match = "inherited"

        confidence *= max(identity.confidence, 0.3)
        return scores, confidence, match


# --------------------------------------------------------------------------- #
# Provider registry
# --------------------------------------------------------------------------- #
_PROVIDERS: list[BenchmarkProvider] = [BuiltinPriorsProvider()]


def register_provider(provider: BenchmarkProvider) -> None:
    """Register an additional benchmark provider (e.g. a network fetcher)."""
    _PROVIDERS.append(provider)


def providers() -> tuple[BenchmarkProvider, ...]:
    return tuple(_PROVIDERS)


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #
@dataclass
class _CacheEntry:
    records: list[BenchmarkRecord]
    fetched_at: int
    raw_ids: list[str] = field(default_factory=list)  # local spellings seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
            "fetched_at": self.fetched_at,
            "raw_ids": sorted(set(self.raw_ids)),
        }


class BenchmarkStore:
    """Cache of benchmark records keyed by canonical model id.

    Lookups are cache-first; a miss consults every registered provider once
    and persists the merged result (via the owner's state persistence — the
    store itself only serializes to/from a plain dict). ``allow_network``
    gates providers that need egress; the builtin table always answers, so
    routing works fully offline.
    """

    def __init__(self, *, allow_network: bool = False, cache_ttl_s: float = 0.0) -> None:
        self.allow_network = allow_network
        self.cache_ttl_s = float(cache_ttl_s)
        self._cache: dict[str, _CacheEntry] = {}
        self.dirty = False  # set when the cache changed and should be persisted

    # -- persistence ------------------------------------------------------ #
    def load_state(self, data: Any) -> None:
        """Load the cache from a previously persisted payload (best-effort)."""
        if not isinstance(data, dict):
            return
        for canonical, body in data.items():
            if not isinstance(body, dict):
                continue
            records = [
                rec
                for raw in body.get("records", []) or []
                if isinstance(raw, dict) and (rec := BenchmarkRecord.from_dict(raw))
            ]
            if not records:
                continue
            self._cache[str(canonical)] = _CacheEntry(
                records=records,
                fetched_at=int(body.get("fetched_at") or 0),
                raw_ids=[str(r) for r in body.get("raw_ids", []) or []],
            )

    def state_payload(self) -> dict[str, Any]:
        return {canonical: e.to_dict() for canonical, e in sorted(self._cache.items())}

    # -- lookups ----------------------------------------------------------- #
    def records_for(self, model_id: str) -> list[BenchmarkRecord]:
        """Records for *model_id*, fetching+caching on first sight."""
        identity = identify(model_id)
        entry = self._cache.get(identity.canonical)
        if entry is not None and not self._expired(entry):
            if identity.raw not in entry.raw_ids:
                entry.raw_ids.append(identity.raw)
                self.dirty = True
            return entry.records
        return self._fetch_and_cache(identity)

    def priors_for(self, model_id: str) -> dict[str, tuple[float, float]]:
        """{capability -> (score, confidence)} for *model_id*.

        Cached records are keyed by the quant-free canonical id, so the
        quantization penalty for this particular local spelling is applied
        here, per lookup."""
        penalty = quant_quality_penalty(identify(model_id).quant)
        out: dict[str, tuple[float, float]] = {}
        for rec in self.records_for(model_id):
            best = out.get(rec.capability)
            # Highest-confidence record wins per capability (exact matches
            # outrank inherited ones).
            if best is None or rec.confidence > best[1]:
                out[rec.capability] = (max(0.0, rec.score - penalty), rec.confidence)
        return out

    def match_confidence(self, model_id: str) -> float:
        """Overall benchmark-match confidence for *model_id* (max over records)."""
        records = self.records_for(model_id)
        return max((r.confidence for r in records), default=0.0)

    # -- management -------------------------------------------------------- #
    def refresh(self, model_ids: Iterable[str] | None = None) -> list[str]:
        """Force re-fetch for *model_ids* (or every cached model). Returns the
        canonical ids refreshed."""
        if model_ids is None:
            canonicals = list(self._cache.keys())
            identities = [identify(c) for c in canonicals]
        else:
            identities = [identify(m) for m in model_ids]
        refreshed = []
        for identity in identities:
            self._cache.pop(identity.canonical, None)
            self._fetch_and_cache(identity)
            refreshed.append(identity.canonical)
        return refreshed

    def clear(self, model_id: str | None = None) -> int:
        """Drop cache entries (all, or just *model_id*'s). Returns count dropped."""
        if model_id is None:
            n = len(self._cache)
            self._cache.clear()
        else:
            n = 1 if self._cache.pop(identify(model_id).canonical, None) else 0
        if n:
            self.dirty = True
        return n

    def summary(self) -> dict[str, Any]:
        """Inspectable cache summary for /admin/benchmarks and /status."""
        return {
            "cached_models": len(self._cache),
            "providers": [
                {"name": p.name, "requires_network": p.requires_network}
                for p in providers()
            ],
            "allow_network": self.allow_network,
            "models": {
                canonical: {
                    "fetched_at": entry.fetched_at,
                    "raw_ids": sorted(set(entry.raw_ids)),
                    "records": [r.to_dict() for r in entry.records],
                }
                for canonical, entry in sorted(self._cache.items())
            },
        }

    # -- internals ---------------------------------------------------------- #
    def _expired(self, entry: _CacheEntry) -> bool:
        if self.cache_ttl_s <= 0:
            return False  # 0 = never expire (refresh is explicit)
        return (time.time() - entry.fetched_at) > self.cache_ttl_s

    def _fetch_and_cache(self, identity: ModelIdentity) -> list[BenchmarkRecord]:
        records: list[BenchmarkRecord] = []
        for provider in providers():
            if provider.requires_network and not self.allow_network:
                continue
            try:
                records.extend(provider.fetch(identity))
            except Exception as exc:  # noqa: BLE001 - providers must not break routing
                log.warning("benchmark provider %s failed for %s: %s",
                            provider.name, identity.canonical, exc)
        self._cache[identity.canonical] = _CacheEntry(
            records=records,
            fetched_at=int(time.time()),
            raw_ids=[identity.raw] if identity.raw else [],
        )
        self.dirty = True
        return records
