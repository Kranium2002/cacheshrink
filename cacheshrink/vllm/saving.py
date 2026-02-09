"""Save cacheshrink MLA models for vLLM serving.

``save_for_vllm()`` creates a vLLM-compatible model directory. It is
completely separate from ``save_mla_model()`` — the HF save path is
untouched.

Output directory contains:
- config.json with architectures=["CacheShrinkForCausalLM"] and
  compressed num_key_value_heads
- Model weights in safetensors format (reuses existing save logic)
- mla_config.json
- Tokenizer files
"""

import json
import os

import torch.nn as nn

from ..saving import save_mla_model


def save_for_vllm(
    model: nn.Module,
    tokenizer,
    save_directory: str,
    training_stats=None,
) -> None:
    """Save an MLA model for vLLM serving.

    Creates a vLLM-compatible directory. Does NOT modify the original model.
    The existing ``save_mla_model()`` is called internally for the actual
    weight/tokenizer saving, then config.json is patched for vLLM.

    Args:
        model: MLA-converted model (must have mla_config attribute)
        tokenizer: HuggingFace tokenizer
        save_directory: Output directory
        training_stats: Optional training stats to include

    Raises:
        ValueError: If model is not MLA-converted, d_latent % d_head != 0,
            or keep_early_layers_original is True
    """
    if not hasattr(model, "mla_config"):
        raise ValueError(
            "Model does not have mla_config attribute. " "Was it converted with convert_to_mla()?"
        )

    mla_config = model.mla_config

    d_latent = mla_config.computed_d_latent
    d_head = mla_config.d_head

    # Validate d_latent is divisible by d_head (required for KV cache shape)
    if d_latent % d_head != 0:
        raise ValueError(
            f"d_latent ({d_latent}) must be divisible by d_head ({d_head}) "
            f"for vLLM KV cache. Current compression_ratio={mla_config.compression_ratio} "
            f"produces d_latent={d_latent}. Try a compression_ratio that gives "
            f"d_latent as a multiple of d_head={d_head}."
        )

    # Validate keep_early_layers_original is not used
    if getattr(mla_config, "keep_early_layers_original", False):
        raise ValueError(
            "keep_early_layers_original=True is not supported in vLLM v1. "
            "All layers must use MLA compression for uniform cache dimensions. "
            "Re-convert with keep_early_layers_original=False."
        )

    # Use save_mla_model for the actual saving (weights, tokenizer, configs)
    save_mla_model(
        model,
        tokenizer,
        save_directory,
        training_stats=training_stats,
        use_safetensors=True,
        enable_hf_loading=True,
    )

    # Now patch config.json for vLLM compatibility
    config_path = os.path.join(save_directory, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"config.json not found in {save_directory}")

    with open(config_path) as f:
        config = json.load(f)

    # Store original architectures and num_key_value_heads
    original_archs = config.get("architectures", [])
    original_n_kv_heads = config.get(
        "num_key_value_heads",
        config.get("n_head", mla_config.n_kv_heads),
    )

    # Set architecture to CacheShrinkForCausalLM so vLLM dispatches to our model
    config["architectures"] = ["CacheShrinkForCausalLM"]

    # Set compressed num_key_value_heads so vLLM allocates smaller KV cache blocks
    n_compressed_kv_heads = d_latent // d_head
    config["num_key_value_heads"] = n_compressed_kv_heads

    # Enrich cacheshrink_mla section with vLLM-specific info
    mla_section = config.get("cacheshrink_mla", mla_config.to_dict())
    # Ensure d_latent is stored as the computed value (not None)
    mla_section["d_latent"] = d_latent
    mla_section["computed_d_latent"] = d_latent
    mla_section["base_architectures"] = original_archs
    mla_section["original_num_kv_heads"] = original_n_kv_heads
    mla_section["n_compressed_kv_heads"] = n_compressed_kv_heads
    mla_section["vllm_compatible"] = True
    config["cacheshrink_mla"] = mla_section

    # Remove auto_map pointing to HF loader (we don't want HF dispatch)
    if "auto_map" in config:
        del config["auto_map"]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Remove the HF modeling stub (not needed for vLLM)
    modeling_stub = os.path.join(save_directory, "modeling_cacheshrink.py")
    if os.path.exists(modeling_stub):
        os.remove(modeling_stub)

    print(f"vLLM model saved to {save_directory}")
    print(f"  Compressed KV heads: {original_n_kv_heads} -> {n_compressed_kv_heads}")
    print(f"  d_latent={d_latent}, d_head={d_head}")
    print(f"  Serve with: vllm serve {save_directory}")
