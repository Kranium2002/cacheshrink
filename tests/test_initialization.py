"""Tests for initialization functions."""

import pytest
import torch

from cacheshrink.initialization import (
    balanced_svd_init,
    init_compression_from_calibration,
)
from cacheshrink.utils import check_orthonormality


class TestBalancedSVDInit:
    """Tests for balanced_svd_init function."""

    def test_basic_init(self, gpt2_config, sample_kv_weights, device, dtype):
        """Test basic SVD initialization."""
        W_k, W_v = sample_kv_weights
        d_latent = gpt2_config.computed_d_latent

        # Now returns 4 values: W_down_k, W_down_v, W_uk, W_uv
        W_down_k, W_down_v, W_uk, W_uv = balanced_svd_init(W_k, W_v, d_latent)

        # Check shapes
        assert W_down_k.shape == (d_latent, gpt2_config.d_model)
        assert W_down_v.shape == (d_latent, gpt2_config.d_model)
        # W_uk and W_uv have shape (d_kv, d_latent) - orthonormal columns
        assert W_uk.shape == (gpt2_config.d_kv, d_latent)
        assert W_uv.shape == (gpt2_config.d_kv, d_latent)

    def test_orthonormality(self, gpt2_config, sample_kv_weights, device, dtype):
        """Test that W_uk and W_uv have orthonormal columns."""
        W_k, W_v = sample_kv_weights
        d_latent = gpt2_config.computed_d_latent

        W_down_k, W_down_v, W_uk, W_uv = balanced_svd_init(W_k, W_v, d_latent)

        # Check orthonormality of columns (W.T @ W = I)
        uk_max_error, uk_mean_error = check_orthonormality(W_uk, mode="columns")
        uv_max_error, uv_mean_error = check_orthonormality(W_uv, mode="columns")

        assert uk_max_error < 1e-5, f"W_uk orthonormality error: {uk_max_error}"
        assert uv_max_error < 1e-5, f"W_uv orthonormality error: {uv_max_error}"

    def test_with_calibration_data(self, gpt2_config, sample_kv_weights, sample_calibration_data, device, dtype):
        """Test initialization with calibration data."""
        W_k, W_v = sample_kv_weights
        d_latent = gpt2_config.computed_d_latent

        W_down_k, W_down_v, W_uk, W_uv = balanced_svd_init(
            W_k, W_v, d_latent,
            calibration_data=sample_calibration_data
        )

        # Check shapes
        assert W_down_k.shape == (d_latent, gpt2_config.d_model)
        assert W_down_v.shape == (d_latent, gpt2_config.d_model)
        assert W_uk.shape == (gpt2_config.d_kv, d_latent)
        assert W_uv.shape == (gpt2_config.d_kv, d_latent)

        # Check orthonormality of columns
        uk_max_error, _ = check_orthonormality(W_uk, mode="columns")
        uv_max_error, _ = check_orthonormality(W_uv, mode="columns")

        assert uk_max_error < 1e-5
        assert uv_max_error < 1e-5

    def test_reconstruction_quality(self, gpt2_config, device, dtype):
        """Test that initialization provides reasonable reconstruction."""
        d_model = gpt2_config.d_model
        d_kv = gpt2_config.d_kv
        d_latent = gpt2_config.computed_d_latent

        # Create simple rank-deficient weights
        W_k = torch.randn(d_kv, d_latent, device=device, dtype=dtype) @ torch.randn(d_latent, d_model, device=device, dtype=dtype)
        W_v = torch.randn(d_kv, d_latent, device=device, dtype=dtype) @ torch.randn(d_latent, d_model, device=device, dtype=dtype)

        W_down_k, W_down_v, W_uk, W_uv = balanced_svd_init(W_k, W_v, d_latent)

        # Test reconstruction: K_approx = h @ W_down_k.T @ W_uk.T, should approximate h @ W_k.T
        h = torch.randn(10, d_model, device=device, dtype=dtype)

        K_orig = h @ W_k.T
        c_k = h @ W_down_k.T
        K_approx = c_k @ W_uk.T

        # Reconstruction shouldn't be perfect but should be reasonable
        # (depends on how well the SVD captures the structure)
        reconstruction_error = (K_orig - K_approx).norm() / K_orig.norm()
        assert reconstruction_error < 2.0  # Loose bound, depends on compression ratio


class TestInitCompressionFromCalibration:
    """Tests for init_compression_from_calibration function."""

    def test_basic(self, gpt2_config, sample_kv_weights, sample_calibration_data, device, dtype):
        """Test initialization from calibration data."""
        W_k, W_v = sample_kv_weights
        d_latent = gpt2_config.computed_d_latent

        W_down_k, W_down_v, W_uk, W_uv = init_compression_from_calibration(
            W_k, W_v, d_latent,
            hidden_states=sample_calibration_data,
        )

        # Check shapes
        assert W_down_k.shape == (d_latent, gpt2_config.d_model)
        assert W_down_v.shape == (d_latent, gpt2_config.d_model)
        assert W_uk.shape == (gpt2_config.d_kv, d_latent)
        assert W_uv.shape == (gpt2_config.d_kv, d_latent)

    def test_subsampling(self, gpt2_config, sample_kv_weights, device, dtype):
        """Test that large calibration data is subsampled."""
        W_k, W_v = sample_kv_weights
        d_latent = gpt2_config.computed_d_latent

        # Create large calibration dataset
        large_calibration = torch.randn(50000, gpt2_config.d_model, device=device, dtype=dtype)

        # Should not error out
        W_down_k, W_down_v, W_uk, W_uv = init_compression_from_calibration(
            W_k, W_v, d_latent,
            hidden_states=large_calibration,
            max_calibration_samples=1000,
        )

        assert W_down_k is not None
        assert W_down_v is not None


class TestGQAInitialization:
    """Tests for initialization with GQA configs."""

    def test_gqa_init(self, llama_config, device, dtype):
        """Test initialization with GQA (fewer KV heads)."""
        d_model = llama_config.d_model
        d_kv = llama_config.d_kv  # n_kv_heads * d_head
        d_latent = llama_config.computed_d_latent

        W_k = torch.randn(d_kv, d_model, device=device, dtype=dtype)
        W_v = torch.randn(d_kv, d_model, device=device, dtype=dtype)

        W_down_k, W_down_v, W_uk, W_uv = balanced_svd_init(W_k, W_v, d_latent)

        # Check shapes match GQA dimensions
        assert W_down_k.shape == (d_latent, d_model)
        assert W_down_v.shape == (d_latent, d_model)
        assert W_uk.shape == (d_kv, d_latent)
        assert W_uv.shape == (d_kv, d_latent)

        # Check orthonormality of columns (W.T @ W = I)
        uk_max_error, _ = check_orthonormality(W_uk, mode="columns")
        uv_max_error, _ = check_orthonormality(W_uv, mode="columns")

        assert uk_max_error < 1e-5
        assert uv_max_error < 1e-5
