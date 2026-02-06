"""Tests for HuggingFace AutoModelForCausalLM.from_pretrained() integration."""

import json
import os
import tempfile

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("transformers", reason="transformers not installed"),
    reason="transformers required",
)


class TestHFLoadingFileCreation:
    """Tests that save_mla_model creates the right files for HF loading."""

    def test_save_creates_modeling_stub(self, device):
        """save_mla_model() creates modeling_cacheshrink.py when enable_hf_loading=True."""
        from cacheshrink import convert_to_mla, save_mla_model

        model, tokenizer = convert_to_mla(
            "gpt2",
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_mla_model(model, tokenizer, tmpdir)

            stub_path = os.path.join(tmpdir, "modeling_cacheshrink.py")
            assert os.path.exists(stub_path)

            with open(stub_path, "r") as f:
                content = f.read()
            assert "CacheShrinkModelForCausalLM" in content
            assert "cacheshrink.hf_modeling" in content

    def test_save_updates_config_json(self, device):
        """save_mla_model() adds auto_map and cacheshrink_mla to config.json."""
        from cacheshrink import convert_to_mla, save_mla_model

        model, tokenizer = convert_to_mla(
            "gpt2",
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_mla_model(model, tokenizer, tmpdir)

            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)

            # auto_map should be present
            assert "auto_map" in config
            assert "AutoModelForCausalLM" in config["auto_map"]
            assert (
                config["auto_map"]["AutoModelForCausalLM"]
                == "modeling_cacheshrink.CacheShrinkModelForCausalLM"
            )

            # cacheshrink_mla should be present with key fields
            assert "cacheshrink_mla" in config
            mla = config["cacheshrink_mla"]
            assert mla["compression_ratio"] == 4.0
            assert mla["model_type"] == "gpt2"

            # Original model_type should be unchanged
            assert config["model_type"] == "gpt2"

    def test_save_disable_hf_loading(self, device):
        """save_mla_model(enable_hf_loading=False) skips HF file creation."""
        from cacheshrink import convert_to_mla, save_mla_model

        model, tokenizer = convert_to_mla(
            "gpt2",
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_mla_model(model, tokenizer, tmpdir, enable_hf_loading=False)

            # modeling_cacheshrink.py should NOT exist
            stub_path = os.path.join(tmpdir, "modeling_cacheshrink.py")
            assert not os.path.exists(stub_path)

            # config.json should NOT have auto_map
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            assert "auto_map" not in config
            assert "cacheshrink_mla" not in config


class TestHFLoadingRoundTrip:
    """Tests for full round-trip: convert -> save -> AutoModel.from_pretrained()."""

    def test_automodel_loads_successfully(self, device):
        """AutoModelForCausalLM.from_pretrained() loads an MLA model."""
        from transformers import AutoModelForCausalLM

        from cacheshrink import convert_to_mla, save_mla_model

        model, tokenizer = convert_to_mla(
            "gpt2",
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_mla_model(model, tokenizer, tmpdir)

            loaded_model = AutoModelForCausalLM.from_pretrained(
                tmpdir, trust_remote_code=True
            )
            loaded_model.eval()

            assert hasattr(loaded_model, "mla_config")
            assert loaded_model.mla_config.compression_ratio == 4.0

    def test_automodel_output_matches_original(self, device):
        """AutoModel-loaded model produces same logits as original."""
        from transformers import AutoModelForCausalLM

        from cacheshrink import convert_to_mla, save_mla_model

        model, tokenizer = convert_to_mla(
            "gpt2",
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )
        model.eval()

        inputs = tokenizer("Hello world", return_tensors="pt").to(device)

        with torch.no_grad():
            logits_orig = model(**inputs).logits

        with tempfile.TemporaryDirectory() as tmpdir:
            save_mla_model(model, tokenizer, tmpdir)

            loaded_model = AutoModelForCausalLM.from_pretrained(
                tmpdir, trust_remote_code=True
            )
            loaded_model.eval()

            with torch.no_grad():
                logits_loaded = loaded_model(**inputs).logits

            assert torch.allclose(logits_orig, logits_loaded, atol=1e-4), (
                "Logits mismatch between original and AutoModel-loaded model"
            )

    def test_automodel_generation_works(self, device):
        """AutoModel-loaded model can generate text."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from cacheshrink import convert_to_mla, save_mla_model

        model, tokenizer = convert_to_mla(
            "gpt2",
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )

        prompt = "The quick brown fox"

        with tempfile.TemporaryDirectory() as tmpdir:
            save_mla_model(model, tokenizer, tmpdir)

            loaded_model = AutoModelForCausalLM.from_pretrained(
                tmpdir, trust_remote_code=True
            )
            loaded_model.eval()

            loaded_tokenizer = AutoTokenizer.from_pretrained(tmpdir)

            inputs = loaded_tokenizer(prompt, return_tensors="pt").to(
                next(loaded_model.parameters()).device
            )

            with torch.no_grad():
                output = loaded_model.generate(
                    inputs.input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=loaded_tokenizer.eos_token_id,
                )

            text = loaded_tokenizer.decode(output[0], skip_special_tokens=True)
            assert len(text) > len(prompt), "Generation should produce output"


class TestBackwardCompatibility:
    """Tests that load_mla_model() still works after enabling HF loading."""

    def test_load_mla_model_still_works(self, device):
        """load_mla_model() works on a directory with HF loading enabled."""
        from cacheshrink import convert_to_mla, save_mla_model, load_mla_model

        model, tokenizer = convert_to_mla(
            "gpt2",
            compression_ratio=4.0,
            device=device,
            use_calibration=False,
            verbose=False,
        )
        model.eval()

        inputs = tokenizer("Hello world", return_tensors="pt").to(device)

        with torch.no_grad():
            logits_orig = model(**inputs).logits

        with tempfile.TemporaryDirectory() as tmpdir:
            save_mla_model(model, tokenizer, tmpdir)

            loaded_model, loaded_tokenizer = load_mla_model(tmpdir, device=str(device))
            loaded_model.eval()

            with torch.no_grad():
                logits_loaded = loaded_model(**inputs).logits

            assert torch.allclose(logits_orig, logits_loaded, atol=1e-4), (
                "load_mla_model() should still work with HF loading files present"
            )
