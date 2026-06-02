"""Section D — metrics exposition.

render() must produce valid Prometheus text (v0.0.4): histogram TYPE lines with
_bucket/_sum/_count series, the engine_uptime_seconds gauge, and the swap_total
counter. record_* functions must move the rendered numbers. No import of
prometheus_client anywhere in the package.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from router import metrics


def test_render_has_histogram_type_lines():
    text = metrics.render()
    assert "# TYPE swap_duration_seconds histogram" in text
    assert "# TYPE memory_settle_seconds histogram" in text
    assert "# TYPE in_flight_at_swap_start histogram" in text


def test_render_has_bucket_sum_count_lines():
    text = metrics.render()
    # Even with zero observations the structural lines exist.
    assert 'swap_duration_seconds_bucket{le="+Inf"} 0' in text
    assert "swap_duration_seconds_sum 0" in text
    assert "swap_duration_seconds_count 0" in text


def test_render_has_engine_uptime_gauge():
    text = metrics.render()
    assert "# TYPE engine_uptime_seconds gauge" in text


def test_content_type_is_prometheus_004():
    assert metrics.CONTENT_TYPE == "text/plain; version=0.0.4; charset=utf-8"


def test_record_swap_updates_render():
    before = metrics.render()
    assert "swap_duration_seconds_count 0" in before

    metrics.record_swap("ollama", "ds4", 3.5, ok=True)
    after = metrics.render()
    assert "swap_duration_seconds_count 1" in after
    # Counter line present with the right labels + result.
    assert 'swap_total{from="ollama",to="ds4",result="ok"} 1' in after
    # The 3.5s observation lands in the le="5" cumulative bucket (>0.5..<=5).
    assert 'swap_duration_seconds_bucket{le="5"} 1' in after
    assert "swap_duration_seconds_sum 3.5" in after


def test_record_swap_error_result_label():
    metrics.record_swap(None, "ds4", 1.0, ok=False)
    text = metrics.render()
    assert 'swap_total{from="none",to="ds4",result="error"} 1' in text


def test_record_memory_settle_updates_render():
    metrics.record_memory_settle(2.0)
    text = metrics.render()
    assert "memory_settle_seconds_count 1" in text
    assert "memory_settle_seconds_sum 2" in text


def test_record_in_flight_at_swap_updates_render():
    metrics.record_in_flight_at_swap_start(3)
    text = metrics.render()
    assert "in_flight_at_swap_start_count 1" in text


def test_set_active_engine_emits_uptime_series():
    # No active engine -> no uptime series line (only the HELP/TYPE headers).
    assert "engine_uptime_seconds{" not in metrics.render()
    metrics.set_active_engine("ds4")
    text = metrics.render()
    assert 'engine_uptime_seconds{engine="ds4"}' in text
    # Clearing it removes the series again.
    metrics.set_active_engine(None)
    assert "engine_uptime_seconds{" not in metrics.render()


def test_render_ends_with_newline():
    assert metrics.render().endswith("\n")


def test_histogram_buckets_are_cumulative():
    # Two observations in different buckets: cumulative counts must be monotone.
    metrics.record_swap("a", "b", 0.4, ok=True)   # le="0.5"
    metrics.record_swap("a", "b", 50.0, ok=True)  # le="60"
    text = metrics.render()
    lines = [
        ln for ln in text.splitlines() if ln.startswith("swap_duration_seconds_bucket")
    ]
    # Parse cumulative values; they must never decrease.
    vals = [int(ln.rsplit(" ", 1)[1]) for ln in lines]
    assert vals == sorted(vals)
    assert vals[-1] == 2  # +Inf bucket sees both


def test_no_prometheus_client_dependency():
    """The whole point of metrics.py is zero deps: it must not import the
    prometheus_client package, and importing router must not pull it in.

    We parse the AST for actual import statements (the module docstring *does*
    mention the name "prometheus_client" to explain it's deliberately absent)."""
    tree = ast.parse(Path(metrics.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "prometheus_client" not in imported
    # And it isn't imported as a side effect of importing the package.
    assert "prometheus_client" not in sys.modules
