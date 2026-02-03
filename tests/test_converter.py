"""Tests for model conversion."""

import pytest
import torch

# These tests require transformers and may download models
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("transformers", reason="transformers not installed"),
    reason="transformers required"
)


class TestConvertToMLA:
    """Tests for convert_to_mla function."""

    @pytest.fixture
    def small_model_name(self):
        """Use the smallest GPT-2 model for testing."""
        return "gpt2"

    def test_convert_gpt2_basic(self, small_model_name, device):
        """Test basic conversion of GPT-2."""
        from cacheshrink import convert_to_mla

        # Convert with minimal calibration
        model, tokenizer = convert_to_mla(
            small_model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,  # Skip calibration for faster test
            verbose=False,
        )

        # Check model has MLA config
        assert hasattr(model, "mla_config")
        assert model.mla_config.model_type == "gpt2"
        assert model.mla_config.compression_ratio == 4.0

        # Check attention modules were replaced
        attn = model.transformer.h[0].attn
        assert hasattr(attn, "mla")  # GPT2AttentionAdapter wraps MLAAttention

    def test_converted_model_generates(self, small_model_name, device):
        """Test that converted model can generate text."""
        from cacheshrink import convert_to_mla

        model, tokenizer = convert_to_mla(
            small_model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        # Try to generate
        inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Should produce output
        assert outputs.shape[1] > inputs.input_ids.shape[1]

        # Should decode to valid text
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert len(text) > 0

    def test_converted_model_orthonormality(self, small_model_name, device):
        """Test that converted model maintains orthonormality."""
        from cacheshrink import convert_to_mla

        model, tokenizer = convert_to_mla(
            small_model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        # Check orthonormality in all layers
        for layer_idx in range(model.mla_config.n_layers):
            attn = model.transformer.h[layer_idx].attn
            errors = attn.check_orthonormality()

            # Note: float16 computation can have slightly higher errors
            assert errors["W_uk"][0] < 1e-3, f"Layer {layer_idx} W_uk error: {errors['W_uk'][0]}"
            assert errors["W_uv"][0] < 1e-3, f"Layer {layer_idx} W_uv error: {errors['W_uv'][0]}"

    def test_compression_ratio_affects_latent_dim(self, small_model_name, device):
        """Test that different compression ratios produce different latent dimensions."""
        from cacheshrink import convert_to_mla

        model_4x, _ = convert_to_mla(
            small_model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        model_8x, _ = convert_to_mla(
            small_model_name,
            compression_ratio=8.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        assert model_4x.mla_config.computed_d_latent > model_8x.mla_config.computed_d_latent

    def test_explicit_d_latent(self, small_model_name, device):
        """Test explicit d_latent override."""
        from cacheshrink import convert_to_mla

        model, _ = convert_to_mla(
            small_model_name,
            compression_ratio=4.0,
            d_latent=128,  # Explicit override
            device=device,
            use_calibration=False,
            verbose=False,
        )

        assert model.mla_config.computed_d_latent == 128


class TestConversionWithCalibration:
    """Tests for conversion with calibration data."""

    @pytest.fixture
    def small_model_name(self):
        return "gpt2"

    @pytest.mark.slow
    def test_convert_with_calibration(self, small_model_name, device):
        """Test conversion with calibration data collection."""
        from cacheshrink import convert_to_mla

        # Use minimal calibration
        model, tokenizer = convert_to_mla(
            small_model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=True,
            num_calibration_samples=8,  # Very few for speed
            max_calibration_length=64,
            verbose=False,
        )

        assert hasattr(model, "mla_config")

        # Model should still generate
        inputs = tokenizer("Test", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        assert outputs.shape[1] > inputs.input_ids.shape[1]

    def test_convert_with_custom_texts(self, small_model_name, device):
        """Test conversion with custom calibration texts."""
        from cacheshrink import convert_to_mla

        custom_texts = [
            "This is a test sentence for calibration.",
            "Another sample text that we use for collecting hidden states.",
            "The quick brown fox jumps over the lazy dog.",
        ] * 10  # Repeat to have enough data

        model, tokenizer = convert_to_mla(
            small_model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=True,
            calibration_texts=custom_texts,
            max_calibration_length=32,
            verbose=False,
        )

        assert hasattr(model, "mla_config")
