"""Integration tests for the full MLA pipeline."""

import os
import tempfile

import pytest
import torch

# These tests require transformers
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("transformers", reason="transformers not installed"),
    reason="transformers required"
)


class TestFullPipeline:
    """End-to-end integration tests."""

    @pytest.fixture
    def model_name(self):
        return "gpt2"

    def test_convert_generate_save_load(self, model_name, device):
        """Test full pipeline: convert -> generate -> save -> load -> generate."""
        from cacheshrink import convert_to_mla, save_mla_model, load_mla_model

        # Convert
        model, tokenizer = convert_to_mla(
            model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )
        model.eval()

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_mla_model(model, tokenizer, tmpdir)

            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, "mla_config.json"))
            assert os.path.exists(os.path.join(tmpdir, "mla_model_marker.json"))
            assert (
                os.path.exists(os.path.join(tmpdir, "model.safetensors")) or
                os.path.exists(os.path.join(tmpdir, "pytorch_model.bin"))
            )

            # Load
            loaded_model, loaded_tokenizer = load_mla_model(tmpdir, device=device)
            loaded_model.eval()

            # Verify model structure matches by checking parameter shapes and values
            orig_params = dict(model.named_parameters())
            loaded_params = dict(loaded_model.named_parameters())

            # Check that all original params exist in loaded model with matching values
            for name, param in orig_params.items():
                assert name in loaded_params, f"Missing parameter: {name}"
                assert param.shape == loaded_params[name].shape, f"Shape mismatch for {name}"
                assert torch.allclose(param.float(), loaded_params[name].float(), atol=1e-5), f"Value mismatch for {name}"

            # Test forward pass produces same logits
            prompt = "Hello world"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                logits1 = model(**inputs).logits
                logits2 = loaded_model(**inputs).logits

            assert torch.allclose(logits1, logits2, atol=1e-4), "Logits mismatch after save/load"

            # Test generation works (output may vary slightly due to numerics, so just check it runs)
            with torch.no_grad():
                output = loaded_model.generate(
                    inputs.input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=loaded_tokenizer.eos_token_id,
                )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            assert len(text) > len(prompt), "Generation should produce output"

    def test_kv_cache_compression(self, model_name, device):
        """Test that KV cache is actually compressed."""
        from cacheshrink import convert_to_mla, measure_cache_memory

        model, tokenizer = convert_to_mla(
            model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        # Measure cache
        stats = measure_cache_memory(model, sequence_lengths=[128, 512])

        # Check compression ratio
        assert stats["compression_ratio"] >= 3.5  # Should be close to 4x

        # Check per-sequence stats
        for seq_len, seq_stats in stats["per_sequence_length"].items():
            assert seq_stats["mla_cache_bytes"] < seq_stats["standard_cache_bytes"]
            assert seq_stats["compression_ratio"] >= 3.5


class TestOrthonormalityPreservation:
    """Tests for orthonormality preservation."""

    @pytest.fixture
    def model_name(self):
        return "gpt2"

    def test_orthonormality_after_forward(self, model_name, device):
        """Test orthonormality is maintained after forward passes."""
        from cacheshrink import convert_to_mla

        model, tokenizer = convert_to_mla(
            model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        # Do some forward passes
        texts = [
            "Hello, how are you?",
            "The quick brown fox",
            "Machine learning is fascinating",
        ]

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model(**inputs)

        # Check orthonormality is still good
        for layer_idx in range(model.mla_config.n_layers):
            attn = model.transformer.h[layer_idx].attn
            errors = attn.check_orthonormality()

            # Note: float16 computation can have slightly higher errors
            assert errors["W_uk"][0] < 1e-3
            assert errors["W_uv"][0] < 1e-3


class TestEvaluation:
    """Tests for evaluation functions."""

    @pytest.fixture
    def model_name(self):
        return "gpt2"

    def test_compute_perplexity(self, model_name, device):
        """Test perplexity computation."""
        from cacheshrink import convert_to_mla, compute_perplexity

        model, tokenizer = convert_to_mla(
            model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        # Compute perplexity on small dataset
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning has revolutionized many fields.",
            "Natural language processing enables computers to understand text.",
        ]

        ppl = compute_perplexity(
            model, tokenizer, texts,
            batch_size=1,
            max_length=64,
            verbose=False,
        )

        # Perplexity should be a positive finite number
        # Note: With 4x compression and no fine-tuning, perplexity can be quite high
        assert ppl > 0
        assert ppl < float("inf")
        assert not torch.isnan(torch.tensor(ppl))

    def test_generate_samples(self, model_name, device):
        """Test sample generation."""
        from cacheshrink import convert_to_mla, generate_samples

        model, tokenizer = convert_to_mla(
            model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        prompts = [
            "Once upon a time",
            "The future of AI",
        ]

        samples = generate_samples(
            model, tokenizer, prompts,
            max_new_tokens=20,
            temperature=0.0,  # Deterministic
            device=device,
        )

        assert len(samples) == len(prompts)
        for prompt, sample in zip(prompts, samples):
            assert sample.startswith(prompt)
            assert len(sample) > len(prompt)


class TestMLATrainer:
    """Tests for MLATrainer."""

    @pytest.fixture
    def model_name(self):
        return "gpt2"

    def test_trainer_initialization(self, model_name, device):
        """Test trainer can be initialized."""
        from cacheshrink import convert_to_mla, MLATrainer

        model, tokenizer = convert_to_mla(
            model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        trainer = MLATrainer(
            model, tokenizer,
            euclidean_lr=1e-4,
            riemannian_lr=1e-3,
        )

        # Check trainer was set up
        assert trainer.euclidean_optimizer is not None
        assert trainer.riemannian_optimizer is not None

    def test_trainer_freezes_correct_params(self, model_name, device):
        """Test that trainer freezes non-MLA parameters."""
        from cacheshrink import convert_to_mla, MLATrainer

        model, tokenizer = convert_to_mla(
            model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        _ = MLATrainer(model, tokenizer)

        # Check that only MLA params are trainable
        for name, param in model.named_parameters():
            if "mla_compression" in name or "mla." in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    @pytest.mark.slow
    def test_trainer_one_step(self, model_name, device):
        """Test that trainer can do one training step."""
        from cacheshrink import convert_to_mla, MLATrainer

        model, tokenizer = convert_to_mla(
            model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        trainer = MLATrainer(model, tokenizer)

        # Tiny dataset - needs to be long enough for training
        texts = [
            "This is a test sentence that is long enough for the trainer to use. " * 5,
            "Another test sentence that has sufficient length for training purposes. " * 5,
            "The quick brown fox jumps over the lazy dog and runs away. " * 5,
            "Machine learning is a fascinating field of study with many applications. " * 5,
        ]

        # Train for one step
        stats = trainer.train(
            texts,
            num_epochs=1,
            batch_size=2,
            max_length=128,  # Longer max_length to ensure sequences aren't too short
        )

        # Check training happened
        # Note: train_loss is logged every logging_steps, so it might be empty for very short training
        assert trainer.global_step > 0  # At least one step was taken

        # Check orthonormality is still maintained
        for layer_idx in range(model.mla_config.n_layers):
            attn = model.transformer.h[layer_idx].attn
            errors = attn.check_orthonormality()
            # Allow slightly more error after training
            assert errors["W_uk"][0] < 1e-3
            assert errors["W_uv"][0] < 1e-3

    def test_trainer_mutual_exclusivity(self, model_name, device):
        """Test that distillation and reconstruction loss are mutually exclusive."""
        from cacheshrink import convert_to_mla, MLATrainer

        model, tokenizer = convert_to_mla(
            model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            store_original_weights=True,
            verbose=False,
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            MLATrainer(
                model, tokenizer,
                use_distillation=True,
                use_reconstruction_loss=True,
            )

    def test_trainer_reconstruction_loss_requires_original_weights(self, model_name, device):
        """Test that reconstruction loss requires stored original weights."""
        from cacheshrink import convert_to_mla, MLATrainer

        # Convert without storing original weights
        model, tokenizer = convert_to_mla(
            model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            store_original_weights=False,
            verbose=False,
        )

        with pytest.raises(ValueError, match="original weights"):
            MLATrainer(
                model, tokenizer,
                use_distillation=False,
                use_reconstruction_loss=True,
            )

    @pytest.mark.slow
    def test_trainer_reconstruction_loss_one_step(self, model_name, device):
        """Test that trainer with reconstruction loss can do one training step."""
        from cacheshrink import convert_to_mla, MLATrainer

        model, tokenizer = convert_to_mla(
            model_name,
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            store_original_weights=True,
            verbose=False,
        )

        trainer = MLATrainer(
            model, tokenizer,
            use_distillation=False,
            use_reconstruction_loss=True,
            reconstruction_alpha=0.3,
        )

        texts = [
            "This is a test sentence that is long enough for the trainer to use. " * 5,
            "Another test sentence that has sufficient length for training purposes. " * 5,
            "The quick brown fox jumps over the lazy dog and runs away. " * 5,
            "Machine learning is a fascinating field of study with many applications. " * 5,
        ]

        stats = trainer.train(
            texts,
            num_epochs=1,
            batch_size=2,
            max_length=128,
        )

        assert trainer.global_step > 0

        # Check orthonormality is still maintained
        for layer_idx in range(model.mla_config.n_layers):
            attn = model.transformer.h[layer_idx].attn
            errors = attn.check_orthonormality()
            assert errors["W_uk"][0] < 1e-3
            assert errors["W_uv"][0] < 1e-3
