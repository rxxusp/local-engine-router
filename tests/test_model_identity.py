"""Tests for router.model_identity — canonicalizing local model ids.

Covers the spellings the smart picker must unify: GGUF filenames, HF repo ids
(AWQ/GPTQ/EXL2/MLX/fp8 variants), Ollama tags, community fine-tunes, and
abliterated/uncensored variants. All pure-function, no I/O.
"""

from __future__ import annotations

import pytest

from router.model_identity import ModelIdentity, identify, quant_quality_penalty


class TestCommonSpellings:
    def test_gguf_filename(self):
        ident = identify("qwen2.5-7b-instruct-q4_k_m.gguf")
        assert ident.family == "qwen2.5"
        assert ident.size_b == 7.0
        assert ident.variant == "instruct"
        assert ident.quant == "q4_k_m"
        assert ident.fmt == "gguf"
        assert ident.canonical == "qwen2.5-7b-instruct"

    def test_hf_repo_awq(self):
        ident = identify("Qwen/Qwen2.5-7B-Instruct-AWQ")
        assert ident.family == "qwen2.5"
        assert ident.size_b == 7.0
        assert ident.quant == "awq"
        assert ident.fmt == "awq"
        assert ident.canonical == "qwen2.5-7b-instruct"

    def test_same_model_same_canonical_across_formats(self):
        spellings = [
            "qwen2.5-7b-instruct-q4_k_m.gguf",
            "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "Qwen/Qwen2.5-7B-Instruct",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "bartowski/Qwen2.5-7B-Instruct-GGUF",
        ]
        canon = {identify(s).canonical for s in spellings}
        assert canon == {"qwen2.5-7b-instruct"}

    def test_ollama_tag_with_size(self):
        ident = identify("llama3.1:8b")
        assert ident.family == "llama3.1"
        assert ident.size_b == 8.0
        assert ident.canonical == "llama3.1-8b"

    def test_ollama_tag_with_quant(self):
        ident = identify("llama3.1:8b-instruct-q5_K_M")
        assert ident.family == "llama3.1"
        assert ident.size_b == 8.0
        assert ident.variant == "instruct"
        assert ident.quant == "q5_k_m"

    def test_ollama_latest_tag_ignored(self):
        ident = identify("mistral:latest")
        assert ident.family == "mistral"
        assert ident.size_b is None

    def test_meta_llama_hf_repo(self):
        ident = identify("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
        assert ident.family == "llama3.1"
        assert ident.size_b == 8.0
        assert ident.fmt == "gguf"
        assert ident.canonical == "llama3.1-8b-instruct"

    def test_gptq_with_version(self):
        ident = identify("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
        assert ident.family == "mistral"
        assert ident.size_b == 7.0
        assert ident.quant == "gptq"
        assert ident.version == "v0.2"

    def test_mlx_community_format(self):
        ident = identify("mlx-community/Qwen2.5-7B-Instruct-4bit")
        assert ident.fmt == "mlx"
        assert ident.quant == "4bit"

    def test_fp8(self):
        ident = identify("neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8")
        assert ident.family == "llama3.1"
        assert ident.size_b == 70.0
        assert ident.quant == "fp8"

    def test_exl2(self):
        ident = identify("turboderp/Llama-3.1-8B-Instruct-exl2")
        assert ident.quant == "exl2"
        assert ident.fmt == "exl2"

    def test_moe_size(self):
        ident = identify("mixtral-8x7b-instruct-v0.1")
        assert ident.family == "mixtral"
        assert ident.size_b == 56.0

    def test_millions_size(self):
        ident = identify("smollm2:135m")
        assert ident.family == "smollm2"
        assert ident.size_b == pytest.approx(0.135)

    def test_deepseek_r1_distill(self):
        ident = identify("deepseek-r1:14b")
        assert ident.family == "deepseek-r1"
        assert ident.size_b == 14.0

    def test_gpt_oss(self):
        ident = identify("gpt-oss:20b")
        assert ident.family == "gpt-oss"
        assert ident.size_b == 20.0


class TestFineTunesAndAbliterated:
    def test_hermes_fine_tune_inherits_llama_base(self):
        ident = identify("NousResearch/Hermes-3-Llama-3.1-8B")
        assert ident.family == "llama3.1"
        assert ident.fine_tune == "hermes"
        assert ident.base_canonical == "llama3.1-8b-instruct"
        assert ident.confidence < 1.0

    def test_abliterated_variant(self):
        ident = identify("huihui_ai/qwen2.5-abliterated:7b")
        assert ident.family == "qwen2.5"
        assert ident.abliterated is True
        assert ident.base_canonical is not None
        assert "abliterated" in ident.canonical

    def test_uncensored_marks_abliterated(self):
        ident = identify("dolphin-2.9-llama3-8b-uncensored")
        assert ident.abliterated is True
        assert ident.fine_tune == "dolphin"

    def test_plain_model_is_its_own_base(self):
        ident = identify("qwen2.5-7b-instruct")
        assert ident.base_canonical == ident.canonical

    def test_fine_tune_assumed_instruct(self):
        ident = identify("dolphin-llama3.1-8b")
        assert ident.variant == "instruct"


class TestEdgeCases:
    def test_unknown_family_low_confidence(self):
        ident = identify("my-custom-model-Q4_K_M.gguf")
        assert ident.family is None
        assert ident.confidence < 0.5
        assert ident.quant == "q4_k_m"
        # quant / format noise removed from canonical
        assert "q4" not in ident.canonical
        assert "gguf" not in ident.canonical

    def test_empty_id(self):
        ident = identify("")
        assert isinstance(ident, ModelIdentity)
        assert ident.canonical == ""

    def test_no_size_lower_confidence(self):
        full = identify("qwen2.5-7b-instruct")
        bare = identify("qwen2.5-instruct")
        assert bare.confidence < full.confidence

    def test_embedding_detection(self):
        assert identify("nomic-embed-text").is_embedding
        assert identify("bge-m3").is_embedding
        assert not identify("qwen2.5-7b-instruct").is_embedding

    def test_vision_detection(self):
        assert identify("llava:13b").is_vision
        assert identify("qwen2.5-vl-7b").is_vision
        assert not identify("qwen2.5-7b-instruct").is_vision

    def test_never_raises(self):
        for weird in ("///", ":::", "a/b/c/d:e-f_g.gguf", "🦙", "-", "q4_k_m"):
            identify(weird)  # must not raise


class TestQuantPenalty:
    def test_monotone(self):
        assert quant_quality_penalty("q2_k") > quant_quality_penalty("q4_k_m")
        assert quant_quality_penalty("q4_k_m") > quant_quality_penalty("q8_0")
        assert quant_quality_penalty("q8_0") > quant_quality_penalty("fp16")

    def test_unquantized_is_free(self):
        assert quant_quality_penalty(None) == 0.0
        assert quant_quality_penalty("bf16") == 0.0

    def test_awq_counts_as_4bit(self):
        assert quant_quality_penalty("awq") == quant_quality_penalty("q4_k_m")
