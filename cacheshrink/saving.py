"""Save and load MLA-converted models."""

import os
import json
from typing import Tuple, Optional, Union, Dict, Any

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from .config import MLAConfig
from .attention import MLAAttention
from .model_handlers import get_handler, get_attention_adapter


def save_mla_model(
    model: nn.Module,
    tokenizer,
    save_directory: str,
    training_stats: Optional[Dict[str, Any]] = None,
    use_safetensors: bool = True,
) -> None:
    """Save an MLA-converted model.

    Saves:
    - Model weights (safetensors or pytorch format)
    - Tokenizer files
    - Original HuggingFace config
    - MLA config

    Args:
        model: MLA-converted model
        tokenizer: HuggingFace tokenizer
        save_directory: Directory to save to
        training_stats: Optional training statistics to save
        use_safetensors: Whether to use safetensors format (recommended)
    """
    os.makedirs(save_directory, exist_ok=True)

    # Get MLA config
    if not hasattr(model, "mla_config"):
        raise ValueError("Model does not have mla_config attribute. Was it converted with convert_to_mla?")

    mla_config = model.mla_config

    # Save MLA config
    mla_config_path = os.path.join(save_directory, "mla_config.json")
    mla_config.save(mla_config_path)

    # Save original HF config
    if hasattr(model, "config"):
        config_path = os.path.join(save_directory, "config.json")
        model.config.save_pretrained(save_directory)

    # Save tokenizer
    tokenizer.save_pretrained(save_directory)

    # Save model weights
    state_dict = model.state_dict()

    if use_safetensors:
        # Convert any non-tensor values and ensure compatibility
        # Handle shared tensors by cloning them
        cleaned_state_dict = {}
        seen_data_ptrs = {}

        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                # Ensure contiguous and on CPU
                tensor = value.contiguous().cpu()
                data_ptr = tensor.data_ptr()

                # If we've seen this data pointer before, clone the tensor
                if data_ptr in seen_data_ptrs:
                    tensor = tensor.clone()
                else:
                    seen_data_ptrs[data_ptr] = key

                cleaned_state_dict[key] = tensor

        weights_path = os.path.join(save_directory, "model.safetensors")
        save_file(cleaned_state_dict, weights_path)
    else:
        weights_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(state_dict, weights_path)

    # Save training stats if provided
    if training_stats is not None:
        stats_path = os.path.join(save_directory, "training_stats.json")
        with open(stats_path, "w") as f:
            # Convert any tensors to Python types
            serializable_stats = {}
            for key, value in training_stats.items():
                if isinstance(value, torch.Tensor):
                    serializable_stats[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    serializable_stats[key] = [
                        v.tolist() if isinstance(v, torch.Tensor) else v
                        for v in value
                    ]
                else:
                    serializable_stats[key] = value
            json.dump(serializable_stats, f, indent=2)

    # Save a marker file indicating this is an MLA model
    marker_path = os.path.join(save_directory, "mla_model_marker.json")
    with open(marker_path, "w") as f:
        json.dump({
            "format_version": "1.0",
            "model_type": mla_config.model_type,
            "compression_ratio": mla_config.compression_ratio,
            "d_latent": mla_config.computed_d_latent,
        }, f, indent=2)

    print(f"Model saved to {save_directory}")


def load_mla_model(
    load_directory: str,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[nn.Module, "AutoTokenizer"]:
    """Load a saved MLA model.

    Args:
        load_directory: Directory containing saved model
        device: Device to load model to
        dtype: Data type for model

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Check for MLA marker
    marker_path = os.path.join(load_directory, "mla_model_marker.json")
    if not os.path.exists(marker_path):
        raise ValueError(f"Not an MLA model directory: {load_directory}")

    # Load MLA config
    mla_config_path = os.path.join(load_directory, "mla_config.json")
    mla_config = MLAConfig.load(mla_config_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(load_directory, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model architecture (without weights)
    hf_config = AutoConfig.from_pretrained(load_directory, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)

    # Get handler and convert architecture to MLA
    handler = get_handler(model, mla_config)
    adapter_class = get_attention_adapter(mla_config.model_type)

    for layer_idx in range(mla_config.n_layers):
        # Create MLA attention
        mla_attn = MLAAttention(mla_config, layer_idx=layer_idx)
        mla_attn = mla_attn.to(device)
        if dtype is not None:
            mla_attn = mla_attn.to(dtype)

        # Wrap with adapter
        if mla_config.model_type == "gpt2":
            adapted_attn = adapter_class(mla_attn)
        else:
            adapted_attn = adapter_class(mla_attn, mla_config)

        # Replace in model
        handler.replace_attention(layer_idx, adapted_attn)

    # Load weights
    safetensors_path = os.path.join(load_directory, "model.safetensors")
    pytorch_path = os.path.join(load_directory, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=device)
    else:
        raise ValueError(f"No model weights found in {load_directory}")

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys in state dict: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state dict: {unexpected_keys[:5]}...")

    # Attach config
    model.mla_config = mla_config

    print(f"Model loaded from {load_directory}")

    return model, tokenizer


def load_training_stats(load_directory: str) -> Optional[Dict[str, Any]]:
    """Load training statistics if available.

    Args:
        load_directory: Directory containing saved model

    Returns:
        Training statistics dict or None if not found
    """
    stats_path = os.path.join(load_directory, "training_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            return json.load(f)
    return None
