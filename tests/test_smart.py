"""Unit tests for router.smart — classifier, scoring, swap-aware picking,
eligibility, health/cooldown, and state persistence.

All hermetic: fake engines from conftest, no network, no GPU.
"""

from __future__ import annotations

import contextlib
import os
import time

import pytest
from conftest import FakeAPISwapEngine, FakeEngine, make_config, make_manager_with_fakes

from router.config import ModelSpec
from router.engines import EngineManager
from router.smart import (
    SmartRouter,
    classify_request,
    is_cloud_model,
)

CHAT = "/v1/chat/completions"


@pytest.fixture(autouse=True)
def _fresh_state_file():
    """EngineManager loads smart health/benchmark state from the (shared)
    default test state file at construction; remove it around each test so
    picks here are hermetic and order-independent."""
    path = "/tmp/local-engine-router-test-state.json"
    with contextlib.suppress(OSError):
        os.unlink(path)
    yield
    with contextlib.suppress(OSError):
        os.unlink(path)


def _msg(text: str, **body):
    return {"model": "smart", "messages": [{"role": "user", "content": text}], **body}


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class TestClassifier:
    def test_plain_chat_is_general(self):
        job = classify_request(CHAT, _msg("hello, how are you today?"))
        assert max(job, key=job.get) == "general"

    def test_code_fence_boosts_coding(self):
        job = classify_request(CHAT, _msg("why does this fail?\n```python\ndef f():\n  pass\n```"))
        assert job.get("coding", 0) > job.get("writing", 0)
        assert job.get("coding", 0) > 0.15

    def test_stack_trace_boosts_coding(self):
        job = classify_request(
            CHAT,
            _msg('Traceback (most recent call last)\n  File "app.py", line 3\nKeyError'),
        )
        assert job.get("coding", 0) > 0.1

    def test_diff_boosts_code_editing(self):
        job = classify_request(
            CHAT, _msg("apply this\n--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n-a\n+b")
        )
        assert job.get("code_editing", 0) > 0.15

    def test_math_prompt(self):
        job = classify_request(
            CHAT, _msg("Solve the equation 3x + 5 = 20 and prove your answer step by step.")
        )
        assert job.get("math", 0) > 0.1
        assert job.get("reasoning", 0) > 0.05

    def test_tools_boost_tool_use(self):
        body = _msg("book a flight", tools=[{"type": "function", "function": {"name": "f"}}])
        job = classify_request(CHAT, body)
        assert max(job, key=job.get) == "tool_use"

    def test_json_response_format(self):
        body = _msg("list colors", response_format={"type": "json_object"})
        job = classify_request(CHAT, body)
        assert job.get("json_structured", 0) > 0.2

    def test_ollama_format_json(self):
        job = classify_request("/api/chat", _msg("list colors", format="json"))
        assert job.get("json_structured", 0) > 0.2

    def test_summarization_prompt(self):
        job = classify_request(CHAT, _msg("Summarize this article: ..."))
        assert job.get("summarization", 0) > 0.1

    def test_writing_prompt(self):
        job = classify_request(CHAT, _msg("Write a story about a lighthouse keeper."))
        assert job.get("writing", 0) > 0.1

    def test_long_context(self):
        job = classify_request(CHAT, _msg("x" * 200_000))
        assert job.get("long_context", 0) > 0.1

    def test_embeddings_endpoint(self):
        job = classify_request("/v1/embeddings", {"model": "smart", "input": "hi"})
        assert job == {"embedding": 1.0}

    def test_small_budget_boosts_speed(self):
        job = classify_request(CHAT, _msg("hi", max_tokens=50))
        assert job.get("speed", 0) > 0

    def test_weights_normalized(self):
        job = classify_request(CHAT, _msg("```py\nx=1\n``` summarize and solve 2+2"))
        assert abs(sum(job.values()) - 1.0) < 0.01

    def test_deterministic(self):
        body = _msg("fix this ```python\nx=\n``` please")
        assert classify_request(CHAT, body) == classify_request(CHAT, body)


class TestCloudDetection:
    def test_cloud_names(self):
        for name in ("gpt-4o", "gpt-4o-mini", "gpt-5", "chatgpt-4o-latest",
                     "claude-3-5-sonnet-20241022", "claude-sonnet-4-5",
                     "gemini-2.0-flash", "o3-mini", "grok-3",
                     "text-embedding-3-small", "deepseek-chat"):
            assert is_cloud_model(name), name

    def test_local_names_not_cloud(self):
        for name in ("llama3.1:8b", "qwen2.5-7b-instruct-q4_k_m.gguf",
                     "mistral:latest", "my-model", "deepseek-r1:14b"):
            assert not is_cloud_model(name), name


# --------------------------------------------------------------------------- #
# Picker fixtures
# --------------------------------------------------------------------------- #
def _two_engine_manager(
    *, active: str | None = None, models: list[ModelSpec] | None = None, **cfg_over
) -> EngineManager:
    models = models if models is not None else [
        ModelSpec(id="llama3.1-8b-instruct", engine="fast_engine",
                  display_name="Llama 8B", context_length=131072),
        ModelSpec(id="llama3.1-70b-instruct", engine="big_engine",
                  display_name="Llama 70B", context_length=131072),
    ]
    cfg = make_config(models=models, **cfg_over)
    mgr = make_manager_with_fakes(
        {
            "fast_engine": FakeEngine("fast_engine", ready=active == "fast_engine"),
            "big_engine": FakeEngine("big_engine", ready=active == "big_engine"),
        },
        cfg=cfg,
    )
    mgr.active_engine = active
    # make_manager_with_fakes swaps the engine table after construction;
    # rebuild the picker's view of the world is not needed (it reads
    # manager.engines dynamically) but the index must match the cfg models.
    return mgr


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #
class TestEligibility:
    async def test_manual_mode_never_picks(self):
        mgr = _two_engine_manager(routing_mode="manual")
        assert await mgr.smart.maybe_pick("smart", CHAT, _msg("hi")) is None

    async def test_exact_model_id_routes_exactly(self):
        mgr = _two_engine_manager()
        assert await mgr.smart.maybe_pick(
            "llama3.1-8b-instruct", CHAT, _msg("hi")
        ) is None

    async def test_configured_alias_routes_exactly(self):
        mgr = _two_engine_manager(aliases={"prod": "llama3.1-8b-instruct"})
        assert await mgr.smart.maybe_pick("prod", CHAT, _msg("hi")) is None

    async def test_smart_alias_triggers_pick(self):
        mgr = _two_engine_manager(active="fast_engine")
        decision = await mgr.smart.maybe_pick("smart", CHAT, _msg("hi"))
        assert decision is not None
        assert decision.model in ("llama3.1-8b-instruct", "llama3.1-70b-instruct")

    async def test_cloud_model_triggers_pick(self):
        mgr = _two_engine_manager(active="fast_engine")
        decision = await mgr.smart.maybe_pick("gpt-4o-mini", CHAT, _msg("hi"))
        assert decision is not None
        assert decision.requested_model == "gpt-4o-mini"

    async def test_unknown_model_triggers_pick(self):
        mgr = _two_engine_manager(active="fast_engine")
        decision = await mgr.smart.maybe_pick("mystery-model-9000", CHAT, _msg("hi"))
        assert decision is not None

    async def test_live_tag_is_exact(self):
        cfg = make_config(models=[])
        mgr = make_manager_with_fakes(
            {"oll": FakeAPISwapEngine("oll", tags={"pulled-model:latest"})}, cfg=cfg
        )
        assert await mgr.smart.maybe_pick(
            "pulled-model:latest", CHAT, _msg("hi")
        ) is None

    async def test_override_exact_model_ids(self):
        mgr = _two_engine_manager(active="fast_engine")
        mgr.smart.scfg.override_exact_model_ids = True
        decision = await mgr.smart.maybe_pick(
            "llama3.1-8b-instruct", CHAT, _msg("hi")
        )
        assert decision is not None

    async def test_catch_flags_off_restores_legacy(self):
        mgr = _two_engine_manager()
        mgr.smart.scfg.catch_cloud_models = False
        mgr.smart.scfg.catch_unknown_models = False
        assert await mgr.smart.maybe_pick("gpt-4o", CHAT, _msg("hi")) is None
        assert await mgr.smart.maybe_pick("mystery", CHAT, _msg("hi")) is None

    async def test_no_candidates_falls_back(self):
        cfg = make_config(models=[])
        mgr = make_manager_with_fakes({"e": FakeEngine("e")}, cfg=cfg)
        assert await mgr.smart.maybe_pick("smart", CHAT, _msg("hi")) is None

    async def test_never_raises_on_selection_error(self, monkeypatch):
        mgr = _two_engine_manager()

        async def boom(*a, **k):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(mgr.smart, "_decide", boom)
        assert await mgr.smart.maybe_pick("smart", CHAT, _msg("hi")) is None


# --------------------------------------------------------------------------- #
# Scoring / swap-awareness
# --------------------------------------------------------------------------- #
class TestSwapAwareness:
    async def test_resident_wins_when_quality_gap_small(self):
        # Two same-class models on different engines: stay where we are.
        models = [
            ModelSpec(id="llama3.1-8b-instruct", engine="fast_engine",
                      display_name="A", context_length=131072),
            ModelSpec(id="qwen2.5-7b-instruct", engine="big_engine",
                      display_name="B", context_length=131072),
        ]
        mgr = _two_engine_manager(active="fast_engine", models=models)
        decision = await mgr.smart.maybe_pick("smart", CHAT, _msg("hello there"))
        assert decision is not None
        assert decision.engine == "fast_engine"
        assert decision.would_swap is False

    async def test_stronger_model_wins_when_it_justifies_swap(self):
        models = [
            ModelSpec(id="llama3.2-1b-instruct", engine="fast_engine",
                      display_name="Tiny", context_length=131072),
            ModelSpec(id="llama3.1-70b-instruct", engine="big_engine",
                      display_name="Big", context_length=131072),
        ]
        mgr = _two_engine_manager(active="fast_engine", models=models)
        mgr.smart.scfg.policy = "quality"
        decision = await mgr.smart.maybe_pick(
            "smart", CHAT, _msg("Prove the theorem step by step with rigor.")
        )
        assert decision is not None
        assert decision.model == "llama3.1-70b-instruct"
        assert decision.would_swap is True
        assert decision.swap_cost_s > 0

    async def test_specialized_model_wins_its_specialty_same_engine(self):
        # Same engine for both -> no swap penalty; the coder must win coding.
        models = [
            ModelSpec(id="qwen2.5-coder-14b-instruct", engine="fast_engine",
                      display_name="Coder", context_length=131072),
            ModelSpec(id="qwen2.5-14b-instruct", engine="fast_engine",
                      display_name="Chat", context_length=131072),
        ]
        mgr = _two_engine_manager(active="fast_engine", models=models)
        decision = await mgr.smart.maybe_pick(
            "smart", CHAT,
            _msg("fix this bug\n```python\ndef f(:\n```\nTraceback (most recent call last)"),
        )
        assert decision is not None
        assert decision.model == "qwen2.5-coder-14b-instruct"

        prose = await mgr.smart.maybe_pick(
            "smart", CHAT, _msg("Write a story about a quiet harbor town.")
        )
        assert prose is not None
        assert prose.model == "qwen2.5-14b-instruct"

    async def test_swap_margin_configurable(self):
        models = [
            ModelSpec(id="llama3.2-1b-instruct", engine="fast_engine",
                      display_name="Tiny", context_length=131072),
            ModelSpec(id="llama3.1-70b-instruct", engine="big_engine",
                      display_name="Big", context_length=131072),
        ]
        mgr = _two_engine_manager(active="fast_engine", models=models)
        mgr.smart.scfg.policy = "quality"
        mgr.smart.scfg.swap_margin = 0.99  # nothing can ever justify a swap
        decision = await mgr.smart.maybe_pick("smart", CHAT, _msg("deep question"))
        assert decision is not None
        assert decision.engine == "fast_engine"
        assert any("swap margin" in r for r in decision.reasons)

    async def test_observed_swap_duration_feeds_estimate(self):
        mgr = _two_engine_manager(active="fast_engine")
        assert mgr.smart.swap_cost_estimate("big_engine") > 0
        mgr._record_swap("fast_engine", "big_engine", 42.0, True)
        est = mgr.smart.swap_cost_estimate("big_engine")
        assert 40.0 <= est <= 45.0
        # Active engine always costs nothing.
        assert mgr.smart.swap_cost_estimate("fast_engine") == 0.0

    async def test_context_overflow_excluded(self):
        models = [
            ModelSpec(id="llama3.1-8b-instruct", engine="fast_engine",
                      display_name="Small ctx", context_length=2048),
            ModelSpec(id="llama3.1-70b-instruct", engine="big_engine",
                      display_name="Big ctx", context_length=131072),
        ]
        mgr = _two_engine_manager(active="fast_engine", models=models)
        decision = await mgr.smart.maybe_pick("smart", CHAT, _msg("x" * 40_000))
        assert decision is not None
        assert decision.model == "llama3.1-70b-instruct"
        excluded = [c for c in decision.candidates if c.excluded]
        assert any("context" in c.excluded for c in excluded)


class TestEmbeddingsAndVision:
    async def test_embeddings_pick_embedding_model(self):
        models = [
            ModelSpec(id="llama3.1-8b-instruct", engine="fast_engine",
                      display_name="Chat", context_length=131072),
            ModelSpec(id="nomic-embed-text", engine="big_engine",
                      display_name="Embed", context_length=8192,
                      capabilities=["embedding"]),
        ]
        mgr = _two_engine_manager(active="fast_engine", models=models)
        decision = await mgr.smart.maybe_pick(
            "smart", "/v1/embeddings", {"model": "smart", "input": "hello"}
        )
        assert decision is not None
        assert decision.model == "nomic-embed-text"

    async def test_no_embedding_candidate_falls_back(self):
        mgr = _two_engine_manager()
        decision = await mgr.smart.maybe_pick(
            "smart", "/v1/embeddings", {"model": "smart", "input": "hello"}
        )
        assert decision is None  # legacy routing handles it

    async def test_embedding_model_excluded_from_chat(self):
        models = [
            ModelSpec(id="nomic-embed-text", engine="fast_engine",
                      display_name="Embed", context_length=8192,
                      capabilities=["embedding"]),
            ModelSpec(id="llama3.1-8b-instruct", engine="big_engine",
                      display_name="Chat", context_length=131072),
        ]
        mgr = _two_engine_manager(active="fast_engine", models=models)
        decision = await mgr.smart.maybe_pick("smart", CHAT, _msg("hi"))
        assert decision is not None
        assert decision.model == "llama3.1-8b-instruct"

    async def test_vision_request_requires_vision_model(self):
        models = [
            ModelSpec(id="llama3.1-8b-instruct", engine="fast_engine",
                      display_name="Chat", context_length=131072),
            ModelSpec(id="llava:13b", engine="big_engine",
                      display_name="Vision", context_length=32768,
                      capabilities=["vision"]),
        ]
        mgr = _two_engine_manager(active="fast_engine", models=models)
        body = {
            "model": "smart",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }],
        }
        decision = await mgr.smart.maybe_pick("smart", CHAT, body)
        assert decision is not None
        assert decision.model == "llava:13b"


# --------------------------------------------------------------------------- #
# Metadata overrides
# --------------------------------------------------------------------------- #
class TestMetadata:
    async def test_strengths_override_beats_priors(self):
        models = [
            ModelSpec(id="mystery-a", engine="fast_engine", display_name="A",
                      context_length=32768, strengths={"coding": 0.95}),
            ModelSpec(id="mystery-b", engine="fast_engine", display_name="B",
                      context_length=32768),
        ]
        mgr = _two_engine_manager(active="fast_engine", models=models)
        decision = await mgr.smart.maybe_pick(
            "smart", CHAT, _msg("```python\ndef f():\n``` fix this bug please")
        )
        assert decision is not None
        assert decision.model == "mystery-a"

    async def test_smart_enabled_false_excludes_model(self):
        models = [
            ModelSpec(id="llama3.1-70b-instruct", engine="fast_engine",
                      display_name="Hidden", context_length=131072,
                      smart_enabled=False),
            ModelSpec(id="llama3.2-1b-instruct", engine="fast_engine",
                      display_name="Tiny", context_length=131072),
        ]
        mgr = _two_engine_manager(active="fast_engine", models=models)
        decision = await mgr.smart.maybe_pick("smart", CHAT, _msg("hi"))
        assert decision is not None
        assert decision.model == "llama3.2-1b-instruct"

    async def test_quality_tier_used_without_priors(self):
        models = [
            ModelSpec(id="mystery-good", engine="fast_engine", display_name="G",
                      context_length=32768, quality_tier=5),
            ModelSpec(id="mystery-bad", engine="fast_engine", display_name="B",
                      context_length=32768, quality_tier=1),
        ]
        mgr = _two_engine_manager(active="fast_engine", models=models)
        decision = await mgr.smart.maybe_pick("smart", CHAT, _msg("hi"))
        assert decision is not None
        assert decision.model == "mystery-good"


# --------------------------------------------------------------------------- #
# Health / cooldown
# --------------------------------------------------------------------------- #
class TestHealth:
    async def test_cooldown_after_threshold_failures(self):
        mgr = _two_engine_manager(active="fast_engine")
        smart = mgr.smart
        for _ in range(smart.scfg.retry.failure_threshold):
            smart.record_failure("llama3.1-70b-instruct", "connect refused")
        assert smart.in_cooldown("llama3.1-70b-instruct")
        decision = await smart.maybe_pick("smart", CHAT, _msg("hi"))
        assert decision is not None
        assert decision.model == "llama3.1-8b-instruct"
        excluded = [c for c in decision.candidates if c.excluded]
        assert any("cooldown" in c.excluded for c in excluded)

    async def test_cooldown_expires(self):
        mgr = _two_engine_manager()
        smart = mgr.smart
        for _ in range(3):
            smart.record_failure("m", "boom")
        smart.health["m"].cooldown_until = time.time() - 1
        assert not smart.in_cooldown("m")

    async def test_success_resets(self):
        mgr = _two_engine_manager()
        smart = mgr.smart
        for _ in range(3):
            smart.record_failure("m", "boom")
        smart.record_success("m")
        assert not smart.in_cooldown("m")
        assert smart.health["m"].consecutive_failures == 0

    async def test_all_cooled_down_still_serves(self):
        # Every candidate in cooldown: the picker must degrade gracefully and
        # still pick something rather than failing the request.
        mgr = _two_engine_manager(active="fast_engine")
        for m in ("llama3.1-8b-instruct", "llama3.1-70b-instruct"):
            for _ in range(3):
                mgr.smart.record_failure(m, "boom")
        decision = await mgr.smart.maybe_pick("smart", CHAT, _msg("hi"))
        assert decision is not None
        assert decision.model  # picked one anyway


# --------------------------------------------------------------------------- #
# Decision payload / state
# --------------------------------------------------------------------------- #
class TestDecisionPayload:
    async def test_decision_dict_is_complete(self):
        mgr = _two_engine_manager(active="fast_engine")
        decision = await mgr.smart.maybe_pick("smart", CHAT, _msg("hi"))
        out = decision.to_dict()
        for key in ("mode", "requested_model", "model", "engine", "policy",
                    "confidence", "job", "would_swap", "swap_cost_s",
                    "reasons", "candidates", "fallbacks", "benchmarks", "retry"):
            assert key in out, key
        assert out["retry"]["never_after_streamed_bytes"] is True
        assert 0.0 <= out["confidence"] <= 1.0
        # candidate breakdown carries score components
        scored = [c for c in out["candidates"] if not c.get("excluded")]
        assert scored and "components" in scored[0]
        assert "quality" in scored[0]["components"]

    async def test_fallbacks_ranked_and_bounded(self):
        models = [
            ModelSpec(id=f"llama3.1-{s}-instruct", engine="fast_engine",
                      display_name=s, context_length=131072)
            for s in ("8b", "70b")
        ] + [
            ModelSpec(id="qwen2.5-7b-instruct", engine="big_engine",
                      display_name="q", context_length=131072),
        ]
        mgr = _two_engine_manager(active="fast_engine", models=models)
        mgr.smart.scfg.retry.max_fallbacks = 1
        decision = await mgr.smart.maybe_pick("smart", CHAT, _msg("hi"))
        assert len(decision.fallbacks) == 1
        assert decision.model not in decision.fallbacks

    async def test_benchmark_provenance_attached(self):
        mgr = _two_engine_manager(active="fast_engine")
        decision = await mgr.smart.maybe_pick("smart", CHAT, _msg("hello"))
        assert decision.benchmarks
        rec = decision.benchmarks[0]
        assert rec["source"] == "builtin-priors"
        assert rec["benchmark"]

    async def test_last_pick_recorded(self):
        mgr = _two_engine_manager(active="fast_engine")
        await mgr.smart.maybe_pick("smart", CHAT, _msg("hi"))
        assert mgr.smart.last_pick is not None
        assert mgr.smart.last_pick["requested_model"] == "smart"


class TestStatePersistence:
    def test_state_round_trip(self):
        mgr = _two_engine_manager()
        smart = mgr.smart
        smart.record_failure("m", "boom")
        smart.note_swap("big_engine", 33.0, ok=True)
        smart.calibration["m"] = {"scores": {"math": 0.85}, "tokens_per_s": 42.0}
        payload = smart.state_payload()

        fresh = SmartRouter(mgr.cfg, mgr)
        fresh.load_state(payload)
        assert fresh.health["m"].total_failures == 1
        assert fresh._swap_seconds["big_engine"] == 33.0
        assert fresh.calibration["m"]["tokens_per_s"] == 42.0

    def test_load_state_tolerates_garbage(self):
        mgr = _two_engine_manager()
        smart = SmartRouter(mgr.cfg, mgr)
        smart.load_state(None)
        smart.load_state("junk")
        smart.load_state({"health": {"m": "junk"}, "swap_seconds": {"e": "x"}})
        assert smart.health["m"].total_failures == 0

    def test_status_summary_shape(self):
        mgr = _two_engine_manager()
        st = mgr.smart.status_summary()
        for key in ("mode", "policy", "aliases", "last_pick", "health",
                    "calibrated_models", "benchmark_cache"):
            assert key in st
