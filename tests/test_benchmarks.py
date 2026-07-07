"""Tests for router.benchmarks — priors, matching, cache, persistence."""

from __future__ import annotations

from router.benchmarks import (
    CAPABILITIES,
    BenchmarkRecord,
    BenchmarkStore,
    BuiltinPriorsProvider,
)
from router.model_identity import identify


def _store() -> BenchmarkStore:
    return BenchmarkStore(allow_network=False, cache_ttl_s=0.0)


class TestBuiltinPriors:
    def test_known_model_covers_all_capabilities(self):
        records = BuiltinPriorsProvider().fetch(identify("qwen2.5-7b-instruct"))
        assert {r.capability for r in records} == set(CAPABILITIES)
        for r in records:
            assert 0.0 <= r.score <= 1.0
            assert 0.0 < r.confidence <= 1.0
            assert r.benchmark  # provenance present
            assert r.url

    def test_bigger_model_scores_higher(self):
        store = _store()
        small = store.priors_for("qwen2.5-1.5b-instruct")
        big = store.priors_for("qwen2.5-72b-instruct")
        assert big["general"][0] > small["general"][0]

    def test_coder_variant_beats_generalist_at_coding(self):
        store = _store()
        coder = store.priors_for("qwen2.5-coder-7b-instruct")
        chat = store.priors_for("qwen2.5-7b-instruct")
        assert coder["coding"][0] > chat["coding"][0]
        assert coder["writing"][0] < chat["writing"][0]

    def test_reasoning_family_boost(self):
        store = _store()
        r1 = store.priors_for("deepseek-r1:14b")
        generic = store.priors_for("qwen2.5:14b")
        assert r1["math"][0] > generic["math"][0]

    def test_quant_penalty_applies(self):
        store = _store()
        full = store.priors_for("Qwen/Qwen2.5-7B-Instruct")
        q2 = store.priors_for("qwen2.5-7b-instruct-q2_k.gguf")
        assert q2["general"][0] < full["general"][0]

    def test_exact_match_outranks_inherited(self):
        store = _store()
        exact_conf = store.match_confidence("qwen2.5-7b-instruct")
        tuned_conf = store.match_confidence("dolphin-qwen2.5-7b")
        assert exact_conf > tuned_conf

    def test_fine_tune_inherits_base_scores(self):
        store = _store()
        tuned = store.priors_for("NousResearch/Hermes-3-Llama-3.1-8B")
        base = store.priors_for("llama3.1-8b-instruct")
        # Same neighbourhood as the base (inherited), just less confident.
        assert abs(tuned["general"][0] - base["general"][0]) < 0.1
        assert tuned["general"][1] < base["general"][1]

    def test_abliterated_small_haircut_not_excluded(self):
        store = _store()
        abl = store.priors_for("huihui_ai/qwen2.5-abliterated:7b")
        base = store.priors_for("qwen2.5:7b")
        assert abl["general"][0] < base["general"][0]
        assert abl["general"][0] > 0.3  # still a usable candidate

    def test_unknown_model_gets_flat_low_confidence_prior(self):
        store = _store()
        priors = store.priors_for("totally-mystery-model")
        assert priors["general"][0] == 0.45
        assert priors["general"][1] <= 0.25


class TestCache:
    def test_fetch_once_per_canonical_model(self):
        store = _store()
        calls = []
        orig = BuiltinPriorsProvider.fetch

        def counting_fetch(self, identity):
            calls.append(identity.canonical)
            return orig(self, identity)

        BuiltinPriorsProvider.fetch = counting_fetch
        try:
            store.records_for("qwen2.5-7b-instruct-q4_k_m.gguf")
            store.records_for("Qwen/Qwen2.5-7B-Instruct-AWQ")  # same canonical
            store.records_for("qwen2.5-7b-instruct")
        finally:
            BuiltinPriorsProvider.fetch = orig
        assert calls == ["qwen2.5-7b-instruct"]

    def test_state_round_trip(self):
        store = _store()
        store.records_for("llama3.1:8b")
        payload = store.state_payload()

        fresh = _store()
        fresh.load_state(payload)
        assert fresh.summary()["cached_models"] == 1
        # Loading from state must not trigger a re-fetch.
        entry = fresh.summary()["models"]["llama3.1-8b"]
        assert entry["records"]
        assert entry["fetched_at"] > 0

    def test_load_state_tolerates_garbage(self):
        store = _store()
        store.load_state(None)
        store.load_state("nonsense")
        store.load_state({"x": {"records": "bad"}, "y": 42})
        assert store.summary()["cached_models"] == 0

    def test_clear_one_and_all(self):
        store = _store()
        store.records_for("llama3.1:8b")
        store.records_for("qwen2.5:7b")
        assert store.clear("llama3.1:8b") == 1
        assert store.summary()["cached_models"] == 1
        assert store.clear() == 1
        assert store.summary()["cached_models"] == 0

    def test_refresh_repopulates(self):
        store = _store()
        store.records_for("llama3.1:8b")
        refreshed = store.refresh(["llama3.1:8b"])
        assert refreshed == ["llama3.1-8b"]
        assert store.summary()["cached_models"] == 1

    def test_dirty_flag_signals_persistence_need(self):
        store = _store()
        assert store.dirty is False
        store.records_for("llama3.1:8b")
        assert store.dirty is True


class TestRecordSerialization:
    def test_round_trip(self):
        rec = BenchmarkRecord(
            source="builtin-priors",
            benchmark="LiveCodeBench",
            capability="coding",
            score=0.61,
            confidence=0.75,
            match="exact",
            canonical_model="qwen2.5-7b-instruct",
            url="https://example.com",
            rank=12,
        )
        back = BenchmarkRecord.from_dict(rec.to_dict())
        assert back is not None
        assert back.capability == "coding"
        assert back.rank == 12

    def test_from_dict_rejects_garbage(self):
        assert BenchmarkRecord.from_dict({}) is None
        assert BenchmarkRecord.from_dict({"source": "x"}) is None
