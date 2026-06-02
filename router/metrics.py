"""Zero-dependency in-process metrics for llm-router.

``prometheus_client`` is intentionally NOT a dependency, so this module
hand-rolls the Prometheus text exposition format (version 0.0.4). It is a tiny
process-global store the EngineManager writes to and a future ``/metrics``
endpoint reads from via :func:`render`.

Exposed series (see the module-level registry below):
  * ``swap_duration_seconds``    histogram  — wall time of a full engine swap
  * ``memory_settle_seconds``    histogram  — time spent waiting for MemAvailable
                                              to plateau after freeing an engine
  * ``in_flight_at_swap_start``  histogram  — in-flight requests being drained
                                              when a swap began
  * ``engine_uptime_seconds``    gauge      — per active engine, computed at
                                              render time from when it last
                                              became active
  * ``swap_total{from,to,result}`` counter  — count of swaps by transition + result

Everything is guarded by a plain ``threading.Lock`` so it is safe to call from
the asyncio event loop (the lock is held only for trivial arithmetic) and from
the (synchronous) render path. ``time.time()`` is used for uptime/timestamps —
this is real runtime code, not a workflow script, so wall-clock is fine.
"""

from __future__ import annotations

import threading
import time
from typing import Iterable

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Histogram bucket upper bounds (seconds / counts). Swaps can take up to ~240s
# for a cold ds4 start; memory settle is ~0-25s; in-flight at swap start is a
# small integer. A single shared layout keeps render() simple and is plenty
# granular for operator dashboards.
_SWAP_BUCKETS: tuple[float, ...] = (
    0.5, 1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 300,
)
_SETTLE_BUCKETS: tuple[float, ...] = (
    0.25, 0.5, 1, 2, 3, 5, 8, 12, 16, 20, 25,
)
_INFLIGHT_BUCKETS: tuple[float, ...] = (
    0, 1, 2, 4, 8, 16, 32, 64,
)


class _Histogram:
    """Cumulative histogram with fixed buckets, a running sum and count."""

    def __init__(self, name: str, help_text: str, buckets: tuple[float, ...]) -> None:
        self.name = name
        self.help = help_text
        self.bounds = tuple(buckets)
        self.counts = [0] * len(self.bounds)  # per-bucket (non-cumulative) tally
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.count += 1
        self.sum += float(value)
        for i, ub in enumerate(self.bounds):
            if value <= ub:
                self.counts[i] += 1
                break
        # values greater than the largest bound still land in +Inf (count/sum).

    def render_lines(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} histogram"
        cumulative = 0
        for ub, c in zip(self.bounds, self.counts):
            cumulative += c
            yield f'{self.name}_bucket{{le="{_fmt(ub)}"}} {cumulative}'
        yield f'{self.name}_bucket{{le="+Inf"}} {self.count}'
        yield f"{self.name}_sum {_fmt(self.sum)}"
        yield f"{self.name}_count {self.count}"


class _Counter:
    """A labelled counter: {label_tuple -> value}."""

    def __init__(self, name: str, help_text: str, label_names: tuple[str, ...]) -> None:
        self.name = name
        self.help = help_text
        self.label_names = label_names
        self.values: dict[tuple[str, ...], float] = {}

    def inc(self, labels: tuple[str, ...], amount: float = 1.0) -> None:
        self.values[labels] = self.values.get(labels, 0.0) + amount

    def render_lines(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} counter"
        for labels, val in sorted(self.values.items()):
            label_str = ",".join(
                f'{n}="{_escape(v)}"' for n, v in zip(self.label_names, labels)
            )
            yield f"{self.name}{{{label_str}}} {_fmt(val)}"


# --------------------------------------------------------------------------- #
# Process-global registry
# --------------------------------------------------------------------------- #
_lock = threading.Lock()

_swap_duration = _Histogram(
    "swap_duration_seconds",
    "Wall-clock duration of a full engine swap, in seconds.",
    _SWAP_BUCKETS,
)
_memory_settle = _Histogram(
    "memory_settle_seconds",
    "Time spent waiting for MemAvailable to plateau after freeing an engine.",
    _SETTLE_BUCKETS,
)
_in_flight_at_swap = _Histogram(
    "in_flight_at_swap_start",
    "Number of in-flight requests being drained when a swap began.",
    _INFLIGHT_BUCKETS,
)
_swap_total = _Counter(
    "swap_total",
    "Total engine swaps by transition and result.",
    ("from", "to", "result"),
)

# Active engine bookkeeping for engine_uptime_seconds. We record the key that
# is currently active and the wall-clock time it became active; uptime is
# derived at render time.
_active_engine: str | None = None
_active_since: float | None = None


# --------------------------------------------------------------------------- #
# Recording API (called by EngineManager)
# --------------------------------------------------------------------------- #
def record_swap(from_key: str | None, to_key: str | None, duration_s: float, ok: bool) -> None:
    """Record a completed swap attempt: duration histogram + result counter."""
    with _lock:
        _swap_duration.observe(duration_s)
        _swap_total.inc(
            (from_key or "none", to_key or "none", "ok" if ok else "error")
        )


def record_memory_settle(seconds: float) -> None:
    """Record how long the post-free memory-settle wait took."""
    with _lock:
        _memory_settle.observe(seconds)


def record_in_flight_at_swap_start(n: int) -> None:
    """Record the number of in-flight requests being drained at swap start."""
    with _lock:
        _in_flight_at_swap.observe(n)


def set_active_engine(key: str | None) -> None:
    """Mark *key* as the currently-active engine (resets its uptime clock)."""
    global _active_engine, _active_since
    with _lock:
        if key == _active_engine:
            return
        _active_engine = key
        _active_since = time.time() if key is not None else None


def reset() -> None:  # pragma: no cover - test/helper convenience
    """Reset all metrics to zero (primarily for tests)."""
    global _active_engine, _active_since
    with _lock:
        for h in (_swap_duration, _memory_settle, _in_flight_at_swap):
            h.counts = [0] * len(h.bounds)
            h.sum = 0.0
            h.count = 0
        _swap_total.values.clear()
        _active_engine = None
        _active_since = None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render() -> str:
    """Render all metrics in Prometheus text exposition format (v0.0.4)."""
    with _lock:
        lines: list[str] = []
        for hist in (_swap_duration, _memory_settle, _in_flight_at_swap):
            lines.extend(hist.render_lines())
        lines.extend(_swap_total.render_lines())

        # engine_uptime_seconds is computed live from _active_since.
        lines.append(
            "# HELP engine_uptime_seconds Seconds the active engine has been "
            "active (since the last successful swap to it)."
        )
        lines.append("# TYPE engine_uptime_seconds gauge")
        if _active_engine is not None and _active_since is not None:
            uptime = max(0.0, time.time() - _active_since)
            lines.append(
                f'engine_uptime_seconds{{engine="{_escape(_active_engine)}"}} '
                f"{_fmt(uptime)}"
            )

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _fmt(value: float) -> str:
    """Format a number for Prometheus exposition (integers stay integral)."""
    f = float(value)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return repr(f)


def _escape(value: str) -> str:
    """Escape a label value per the Prometheus text format."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )
