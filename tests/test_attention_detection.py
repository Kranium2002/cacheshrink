"""Tests for attention type detection."""

import pytest
from unittest.mock import MagicMock

from cacheshrink.attention_detection import (
    AttentionType,
    AttentionInfo,
    detect_attention_type,
    get_compression_method,
    is_gqa_model,
    is_mqa_model,
    is_mha_model,
)


class MockConfig:
    """Mock HuggingFace config for testing."""

    def __init__(
        self,
        num_attention_heads=32,
        num_key_value_heads=None,
        kv_lora_rank=None,
        q_lora_rank=None,
        n_head=None,
    ):
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is not None:
            self.num_key_value_heads = num_key_value_heads
        if kv_lora_rank is not None:
            self.kv_lora_rank = kv_lora_rank
        if q_lora_rank is not None:
            self.q_lora_rank = q_lora_rank
        if n_head is not None:
            self.n_head = n_head


class TestDetectAttentionType:
    """Tests for detect_attention_type function."""

    def test_mha_detection(self):
        """Test MHA detection when n_kv_heads == n_heads."""
        config = MockConfig(num_attention_heads=32, num_key_value_heads=32)
        info = detect_attention_type(config)

        assert info.attention_type == AttentionType.MHA
        assert info.n_heads == 32
        assert info.n_kv_heads == 32
        assert info.recommended_method == "mla"
        assert "MHA" in info.reason

    def test_mha_detection_gpt2_style(self):
        """Test MHA detection for GPT-2 style configs with n_head."""
        config = MockConfig(num_attention_heads=None, n_head=12)
        info = detect_attention_type(config)

        assert info.attention_type == AttentionType.MHA
        assert info.n_heads == 12
        assert info.n_kv_heads == 12
        assert info.recommended_method == "mla"

    def test_mha_default_when_no_kv_heads(self):
        """Test that MHA is assumed when num_key_value_heads is not specified."""
        config = MockConfig(num_attention_heads=32)
        info = detect_attention_type(config)

        assert info.attention_type == AttentionType.MHA
        assert info.n_kv_heads == 32  # Defaults to n_heads

    def test_gqa_detection(self):
        """Test GQA detection when 1 < n_kv_heads < n_heads."""
        config = MockConfig(num_attention_heads=32, num_key_value_heads=8)
        info = detect_attention_type(config)

        assert info.attention_type == AttentionType.GQA
        assert info.n_heads == 32
        assert info.n_kv_heads == 8
        assert info.recommended_method == "xkv"
        assert "GQA" in info.reason or "cross-layer" in info.reason.lower()

    def test_gqa_detection_qwen_style(self):
        """Test GQA detection for Qwen-style config (4 KV heads)."""
        config = MockConfig(num_attention_heads=28, num_key_value_heads=4)
        info = detect_attention_type(config)

        assert info.attention_type == AttentionType.GQA
        assert info.recommended_method == "xkv"

    def test_mqa_detection(self):
        """Test MQA detection when n_kv_heads == 1."""
        config = MockConfig(num_attention_heads=32, num_key_value_heads=1)
        info = detect_attention_type(config)

        assert info.attention_type == AttentionType.MQA
        assert info.n_heads == 32
        assert info.n_kv_heads == 1
        assert info.recommended_method == "xkv"
        assert "MQA" in info.reason

    def test_native_mla_detection_kv_lora(self):
        """Test native MLA detection when kv_lora_rank is present."""
        config = MockConfig(num_attention_heads=32, num_key_value_heads=8, kv_lora_rank=512)
        info = detect_attention_type(config)

        assert info.attention_type == AttentionType.NATIVE_MLA
        assert info.recommended_method == "unsupported"
        assert "native MLA" in info.reason.lower() or "already" in info.reason.lower()

    def test_native_mla_detection_q_lora(self):
        """Test native MLA detection when q_lora_rank is present."""
        config = MockConfig(num_attention_heads=32, num_key_value_heads=8, q_lora_rank=1536)
        info = detect_attention_type(config)

        assert info.attention_type == AttentionType.NATIVE_MLA
        assert info.recommended_method == "unsupported"

    def test_unknown_when_no_heads(self):
        """Test UNKNOWN when num_attention_heads cannot be determined."""
        config = MockConfig(num_attention_heads=None)
        info = detect_attention_type(config)

        assert info.attention_type == AttentionType.UNKNOWN
        assert info.recommended_method == "unsupported"


class TestGetCompressionMethod:
    """Tests for get_compression_method function."""

    def test_auto_returns_mla_for_mha(self):
        """Test that 'auto' returns 'mla' for MHA models."""
        config = MockConfig(num_attention_heads=32, num_key_value_heads=32)
        method = get_compression_method(config, user_method="auto", verbose=False)
        assert method == "mla"

    def test_auto_returns_xkv_for_gqa(self):
        """Test that 'auto' returns 'xkv' for GQA models."""
        config = MockConfig(num_attention_heads=32, num_key_value_heads=8)
        method = get_compression_method(config, user_method="auto", verbose=False)
        assert method == "xkv"

    def test_auto_returns_xkv_for_mqa(self):
        """Test that 'auto' returns 'xkv' for MQA models."""
        config = MockConfig(num_attention_heads=32, num_key_value_heads=1)
        method = get_compression_method(config, user_method="auto", verbose=False)
        assert method == "xkv"

    def test_none_same_as_auto(self):
        """Test that None user_method behaves like 'auto'."""
        config = MockConfig(num_attention_heads=32, num_key_value_heads=8)
        method = get_compression_method(config, user_method=None, verbose=False)
        assert method == "xkv"

    def test_user_override_respected(self):
        """Test that user-specified method is respected (with warning)."""
        config = MockConfig(num_attention_heads=32, num_key_value_heads=8)
        # User forces 'mla' even for GQA
        method = get_compression_method(config, user_method="mla", verbose=False)
        assert method == "mla"

    def test_legacy_methods_respected(self):
        """Test that legacy methods (separate, joint, decoupled_rope) work."""
        config = MockConfig(num_attention_heads=32, num_key_value_heads=32)

        assert get_compression_method(config, "separate", verbose=False) == "separate"
        assert get_compression_method(config, "joint", verbose=False) == "joint"
        assert get_compression_method(config, "decoupled_rope", verbose=False) == "decoupled_rope"

    def test_unsupported_raises_for_native_mla(self):
        """Test that native MLA raises ValueError."""
        config = MockConfig(num_attention_heads=32, kv_lora_rank=512)
        with pytest.raises(ValueError, match="not supported"):
            get_compression_method(config, user_method="auto", verbose=False)

    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        config = MockConfig(num_attention_heads=32, num_key_value_heads=32)
        with pytest.raises(ValueError, match="Unknown compression method"):
            get_compression_method(config, user_method="invalid", verbose=False)


class TestConvenienceFunctions:
    """Tests for is_gqa_model, is_mqa_model, is_mha_model."""

    def test_is_gqa_model(self):
        """Test is_gqa_model returns True for GQA."""
        gqa_config = MockConfig(num_attention_heads=32, num_key_value_heads=8)
        mha_config = MockConfig(num_attention_heads=32, num_key_value_heads=32)
        mqa_config = MockConfig(num_attention_heads=32, num_key_value_heads=1)

        assert is_gqa_model(gqa_config) is True
        assert is_gqa_model(mha_config) is False
        assert is_gqa_model(mqa_config) is False

    def test_is_mqa_model(self):
        """Test is_mqa_model returns True for MQA."""
        gqa_config = MockConfig(num_attention_heads=32, num_key_value_heads=8)
        mha_config = MockConfig(num_attention_heads=32, num_key_value_heads=32)
        mqa_config = MockConfig(num_attention_heads=32, num_key_value_heads=1)

        assert is_mqa_model(gqa_config) is False
        assert is_mqa_model(mha_config) is False
        assert is_mqa_model(mqa_config) is True

    def test_is_mha_model(self):
        """Test is_mha_model returns True for MHA."""
        gqa_config = MockConfig(num_attention_heads=32, num_key_value_heads=8)
        mha_config = MockConfig(num_attention_heads=32, num_key_value_heads=32)
        mqa_config = MockConfig(num_attention_heads=32, num_key_value_heads=1)

        assert is_mha_model(gqa_config) is False
        assert is_mha_model(mha_config) is True
        assert is_mha_model(mqa_config) is False


class TestAttentionInfo:
    """Tests for AttentionInfo dataclass."""

    def test_attention_info_creation(self):
        """Test AttentionInfo dataclass creation."""
        info = AttentionInfo(
            attention_type=AttentionType.GQA,
            n_heads=32,
            n_kv_heads=8,
            recommended_method="xkv",
            reason="Test reason",
        )

        assert info.attention_type == AttentionType.GQA
        assert info.n_heads == 32
        assert info.n_kv_heads == 8
        assert info.recommended_method == "xkv"
        assert info.reason == "Test reason"


class TestAttentionTypeEnum:
    """Tests for AttentionType enum."""

    def test_enum_values(self):
        """Test AttentionType enum has expected values."""
        assert AttentionType.MHA.value == "mha"
        assert AttentionType.GQA.value == "gqa"
        assert AttentionType.MQA.value == "mqa"
        assert AttentionType.NATIVE_MLA.value == "native_mla"
        assert AttentionType.UNKNOWN.value == "unknown"
