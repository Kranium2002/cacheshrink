"""Tests for vLLM integration (cacheshrink.vllm).

Unit tests that do NOT require vLLM to be installed are run unconditionally.
Integration tests requiring vLLM or GPU are marked with pytest.mark.skipif.
"""

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from cacheshrink.config import MLAConfig
from cacheshrink.vllm.config import VLLMMLAConfig
from cacheshrink.vllm.weight_loader import map_weight_name, map_weight_names, is_compression_param


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeHFConfig:
    """Minimal HF config object for testing VLLMMLAConfig extraction."""

    def __init__(self, mla_dict, **extra):
        self.cacheshrink_mla = mla_dict
        self.vocab_size = mla_dict.get("vocab_size", 1000)
        self.intermediate_size = extra.get("intermediate_size", 256)
        self.hidden_act = extra.get("hidden_act", "silu")
        for k, v in extra.items():
            setattr(self, k, v)


def _make_mla_dict(**overrides):
    """Create a minimal cacheshrink_mla dict for testing."""
    base = {
        "model_name": "test-model",
        "model_type": "generic",
        "d_model": 64,
        "n_heads": 4,
        "n_kv_heads": 4,
        "d_head": 16,
        "n_layers": 2,
        "vocab_size": 1000,
        "compression_method": "separate",
        "d_latent": 16,
        "computed_d_latent": 16,
        "use_bias": False,
        "d_rope": 64,
        "compression_ratio": 4.0,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "use_cross_layer": False,
        "cross_layer_group_size": 4,
        "xkv_skip_early_layers": 0,
        "keep_early_layers_original": False,
        "layer_norm_eps": 1e-5,
        "max_position_embeddings": 128,
        "extra_config": {"uses_rope": True},
        "base_architectures": ["LlamaForCausalLM"],
        "original_num_kv_heads": 4,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# VLLMMLAConfig tests
# ---------------------------------------------------------------------------


class TestVLLMMLAConfig:
    """Test VLLMMLAConfig extraction and properties."""

    def test_from_hf_config_basic(self):
        mla_dict = _make_mla_dict()
        hf_config = FakeHFConfig(mla_dict)

        cfg = VLLMMLAConfig.from_hf_config(hf_config)

        assert cfg.d_model == 64
        assert cfg.n_heads == 4
        assert cfg.n_kv_heads == 4
        assert cfg.d_head == 16
        assert cfg.d_latent == 16
        assert cfg.n_layers == 2
        assert cfg.compression_method == "separate"
        assert cfg.uses_rope is True

    def test_from_hf_config_missing_section(self):
        hf_config = type("C", (), {})()
        with pytest.raises(ValueError, match="cacheshrink_mla"):
            VLLMMLAConfig.from_hf_config(hf_config)

    def test_n_compressed_kv_heads(self):
        mla_dict = _make_mla_dict(d_latent=16, d_head=16)
        hf_config = FakeHFConfig(mla_dict)
        cfg = VLLMMLAConfig.from_hf_config(hf_config)
        assert cfg.n_compressed_kv_heads == 1  # 16 // 16

        mla_dict2 = _make_mla_dict(d_latent=32, d_head=16)
        hf_config2 = FakeHFConfig(mla_dict2)
        cfg2 = VLLMMLAConfig.from_hf_config(hf_config2)
        assert cfg2.n_compressed_kv_heads == 2  # 32 // 16

    def test_d_kv_property(self):
        mla_dict = _make_mla_dict(n_kv_heads=8, d_head=16, original_num_kv_heads=8)
        hf_config = FakeHFConfig(mla_dict)
        cfg = VLLMMLAConfig.from_hf_config(hf_config)
        assert cfg.d_kv == 128  # 8 * 16

    def test_to_mla_config(self):
        mla_dict = _make_mla_dict()
        hf_config = FakeHFConfig(mla_dict)
        cfg = VLLMMLAConfig.from_hf_config(hf_config)

        native = cfg.to_mla_config()
        assert isinstance(native, MLAConfig)
        assert native.d_model == cfg.d_model
        assert native.n_heads == cfg.n_heads
        assert native.computed_d_latent == cfg.d_latent
        assert native.n_kv_heads == cfg.n_kv_heads

    def test_xkv_config(self):
        mla_dict = _make_mla_dict(
            use_cross_layer=True,
            cross_layer_group_size=2,
            xkv_skip_early_layers=0,
            n_layers=4,
        )
        hf_config = FakeHFConfig(mla_dict)
        cfg = VLLMMLAConfig.from_hf_config(hf_config)

        assert cfg.use_cross_layer is True
        assert cfg.n_groups == 2  # 4 layers / 2 per group
        assert cfg.is_xkv_layer(0) is True
        assert cfg.get_xkv_group(0) == 0
        assert cfg.get_xkv_group(2) == 1
        assert cfg.get_group_layers(0) == [0, 1]
        assert cfg.get_group_layers(1) == [2, 3]

    def test_xkv_with_skip(self):
        mla_dict = _make_mla_dict(
            use_cross_layer=True,
            cross_layer_group_size=2,
            xkv_skip_early_layers=2,
            n_layers=6,
        )
        hf_config = FakeHFConfig(mla_dict)
        cfg = VLLMMLAConfig.from_hf_config(hf_config)

        assert cfg.is_xkv_layer(0) is False
        assert cfg.is_xkv_layer(1) is False
        assert cfg.is_xkv_layer(2) is True
        assert cfg.get_xkv_group(0) == -1
        assert cfg.get_xkv_group(2) == 0
        assert cfg.n_groups == 2  # (6-2) / 2 = 2


# ---------------------------------------------------------------------------
# Weight name mapping tests
# ---------------------------------------------------------------------------


class TestWeightNameMapping:
    """Test weight name mapping from saved state dict to vLLM model."""

    def _cfg(self):
        mla_dict = _make_mla_dict()
        hf_config = FakeHFConfig(mla_dict)
        return VLLMMLAConfig.from_hf_config(hf_config)

    def test_embed_tokens(self):
        cfg = self._cfg()
        assert (
            map_weight_name("model.model.embed_tokens.weight", cfg)
            == "model.embed_tokens.weight"
        )

    def test_lm_head(self):
        cfg = self._cfg()
        assert map_weight_name("lm_head.weight", cfg) == "lm_head.weight"

    def test_q_proj(self):
        cfg = self._cfg()
        assert (
            map_weight_name(
                "model.model.layers.0.self_attn.mla.q_proj.weight", cfg
            )
            == "model.layers.0.self_attn.q_proj.weight"
        )

    def test_o_proj(self):
        cfg = self._cfg()
        assert (
            map_weight_name(
                "model.model.layers.0.self_attn.mla.o_proj.weight", cfg
            )
            == "model.layers.0.self_attn.o_proj.weight"
        )

    def test_compression_W_down_k(self):
        cfg = self._cfg()
        assert (
            map_weight_name(
                "model.model.layers.0.self_attn.mla.mla_compression.W_down_k.weight",
                cfg,
            )
            == "model.layers.0.self_attn.mla_compression.W_down_k.weight"
        )

    def test_compression_W_uk(self):
        cfg = self._cfg()
        assert (
            map_weight_name(
                "model.model.layers.0.self_attn.mla.mla_compression.W_uk", cfg
            )
            == "model.layers.0.self_attn.mla_compression.W_uk"
        )

    def test_mlp_passthrough(self):
        cfg = self._cfg()
        assert (
            map_weight_name(
                "model.model.layers.0.mlp.gate_proj.weight", cfg
            )
            == "model.layers.0.mlp.gate_proj.weight"
        )

    def test_skip_original_weights(self):
        cfg = self._cfg()
        assert (
            map_weight_name(
                "model.model.layers.0.self_attn.mla.mla_compression.W_k_original",
                cfg,
            )
            is None
        )

    def test_skip_rotary_emb(self):
        cfg = self._cfg()
        assert (
            map_weight_name(
                "model.model.layers.0.self_attn.mla.rotary_emb.inv_freq", cfg
            )
            is None
        )

    def test_skip_xkv_groups(self):
        cfg = self._cfg()
        assert (
            map_weight_name("xkv_groups.0.shared_W_uk", cfg)
            is None
        )

    def test_layernorm(self):
        cfg = self._cfg()
        assert (
            map_weight_name(
                "model.model.layers.0.input_layernorm.weight", cfg
            )
            == "model.layers.0.input_layernorm.weight"
        )

    def test_map_weight_names_batch(self):
        cfg = self._cfg()
        keys = [
            "model.model.layers.0.self_attn.mla.q_proj.weight",
            "model.model.layers.0.self_attn.mla.mla_compression.W_uk",
            "model.model.layers.0.self_attn.mla.mla_compression.W_k_original",
            "lm_head.weight",
        ]
        mapping = map_weight_names(keys, cfg)
        assert len(mapping) == 3  # W_k_original is skipped
        assert "model.model.layers.0.self_attn.mla.mla_compression.W_k_original" not in mapping


# ---------------------------------------------------------------------------
# Compression param detection tests
# ---------------------------------------------------------------------------


class TestCompressionParamDetection:
    """Test is_compression_param detection."""

    def test_W_uk(self):
        assert is_compression_param("model.layers.0.self_attn.mla_compression.W_uk") is True

    def test_W_uv(self):
        assert is_compression_param("model.layers.0.self_attn.mla_compression.W_uv") is True

    def test_W_down_k(self):
        assert is_compression_param("model.layers.0.self_attn.mla_compression.W_down_k.weight") is True

    def test_shared_W_uk(self):
        assert is_compression_param("xkv_groups.0.shared_W_uk") is True

    def test_b_k(self):
        assert is_compression_param("model.layers.0.self_attn.mla_compression.b_k") is True

    def test_q_proj_not_compression(self):
        assert is_compression_param("model.layers.0.self_attn.q_proj.weight") is False

    def test_lm_head_not_compression(self):
        assert is_compression_param("lm_head.weight") is False


# ---------------------------------------------------------------------------
# save_for_vllm validation tests
# ---------------------------------------------------------------------------


class TestSaveForVLLMValidation:
    """Test save_for_vllm input validation (no actual model saving)."""

    def test_no_mla_config_raises(self):
        from cacheshrink.vllm.saving import save_for_vllm

        model = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="mla_config"):
            save_for_vllm(model, None, "/tmp/test")

    def test_d_latent_not_divisible_raises(self):
        from cacheshrink.vllm.saving import save_for_vllm

        model = nn.Linear(10, 10)
        # Create config where d_latent % d_head != 0
        model.mla_config = MLAConfig(
            model_name="test",
            model_type="generic",
            n_heads=4,
            n_kv_heads=4,
            d_model=64,
            d_head=16,
            n_layers=2,
            d_latent=13,  # Not divisible by 16
        )
        with pytest.raises(ValueError, match="divisible"):
            save_for_vllm(model, None, "/tmp/test")

    def test_keep_early_original_raises(self):
        from cacheshrink.vllm.saving import save_for_vllm

        model = nn.Linear(10, 10)
        model.mla_config = MLAConfig(
            model_name="test",
            model_type="generic",
            n_heads=4,
            n_kv_heads=4,
            d_model=64,
            d_head=16,
            n_layers=2,
            d_latent=16,
            keep_early_layers_original=True,
        )
        with pytest.raises(ValueError, match="keep_early_layers_original"):
            save_for_vllm(model, None, "/tmp/test")


# ---------------------------------------------------------------------------
# Compressed KV head count calculation tests
# ---------------------------------------------------------------------------


class TestCompressedKVHeadCount:
    """Test compressed KV head count calculation for various configs."""

    def test_4x_compression_mha(self):
        """MHA model with 4x compression."""
        mla_dict = _make_mla_dict(
            n_kv_heads=4, d_head=16, d_latent=16  # d_kv=64, d_latent=16 -> 4x
        )
        cfg = VLLMMLAConfig.from_hf_config(FakeHFConfig(mla_dict))
        assert cfg.n_compressed_kv_heads == 1  # 16 // 16

    def test_2x_compression_gqa(self):
        """GQA model with 2x compression."""
        mla_dict = _make_mla_dict(
            n_heads=32,
            n_kv_heads=8,
            d_head=128,
            d_latent=512,  # d_kv=1024, d_latent=512 -> 2x
        )
        cfg = VLLMMLAConfig.from_hf_config(FakeHFConfig(mla_dict))
        assert cfg.n_compressed_kv_heads == 4  # 512 // 128

    def test_8x_compression(self):
        """8x compression."""
        mla_dict = _make_mla_dict(
            n_kv_heads=8, d_head=16, d_latent=16  # d_kv=128, d_latent=16
        )
        cfg = VLLMMLAConfig.from_hf_config(FakeHFConfig(mla_dict))
        assert cfg.n_compressed_kv_heads == 1


# ---------------------------------------------------------------------------
# MLAConfig round-trip tests
# ---------------------------------------------------------------------------


class TestMLAConfigRoundTrip:
    """Test VLLMMLAConfig -> MLAConfig -> back."""

    def test_round_trip_preserves_fields(self):
        mla_dict = _make_mla_dict(
            d_model=128,
            n_heads=8,
            n_kv_heads=4,
            d_head=16,
            d_latent=32,
            use_bias=True,
            compression_ratio=2.0,
        )
        hf_config = FakeHFConfig(mla_dict)
        vllm_cfg = VLLMMLAConfig.from_hf_config(hf_config)
        native = vllm_cfg.to_mla_config()

        assert native.d_model == 128
        assert native.n_heads == 8
        assert native.n_kv_heads == 4
        assert native.d_head == 16
        assert native.computed_d_latent == 32
        assert native.use_bias is True
        assert native.compression_ratio == 2.0

    def test_xkv_round_trip(self):
        mla_dict = _make_mla_dict(
            use_cross_layer=True,
            cross_layer_group_size=4,
            xkv_skip_early_layers=2,
        )
        hf_config = FakeHFConfig(mla_dict)
        vllm_cfg = VLLMMLAConfig.from_hf_config(hf_config)
        native = vllm_cfg.to_mla_config()

        assert native.use_cross_layer is True
        assert native.cross_layer_group_size == 4
        assert native.xkv_skip_early_layers == 2
        assert native.keep_early_layers_original is False


# ---------------------------------------------------------------------------
# Integration tests (require vLLM)
# ---------------------------------------------------------------------------

try:
    import vllm  # noqa: F401
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
class TestVLLMPluginRegistration:
    """Test vLLM plugin registration."""

    def test_register_model(self):
        from cacheshrink.vllm.plugin import register_cacheshrink_models

        register_cacheshrink_models()
        from vllm import ModelRegistry

        assert "CacheShrinkForCausalLM" in ModelRegistry.get_supported_archs()
