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
    low_cpu_mem_usage: bool = True,
) -> Tuple[nn.Module, "AutoTokenizer"]:
    """Load a saved MLA model.

    Args:
        load_directory: Directory containing saved model
        device: Device to load model to
        dtype: Data type for model
        low_cpu_mem_usage: Use accelerate for faster loading (default True)

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
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

    # Load base model architecture with empty weights for speed
    hf_config = AutoConfig.from_pretrained(load_directory, trust_remote_code=True)
    
    # Use low_cpu_mem_usage to avoid initializing weights twice
    if low_cpu_mem_usage:
        # Create model with empty weights (much faster)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)

    # Get handler and convert architecture to MLA (on meta device if low_cpu_mem)
    handler = get_handler(model, mla_config)
    adapter_class = get_attention_adapter(mla_config.model_type)

    for layer_idx in range(mla_config.n_layers):
        # Create MLA attention
        if low_cpu_mem_usage:
            with torch.device("meta"):
                mla_attn = MLAAttention(mla_config, layer_idx=layer_idx)
        else:
            mla_attn = MLAAttention(mla_config, layer_idx=layer_idx)

        # Wrap with adapter
        if mla_config.model_type == "gpt2":
            adapted_attn = adapter_class(mla_attn)
        else:
            adapted_attn = adapter_class(mla_attn, mla_config)

        # Replace in model
        handler.replace_attention(layer_idx, adapted_attn)

    # Load weights directly to device
    safetensors_path = os.path.join(load_directory, "model.safetensors")
    pytorch_path = os.path.join(load_directory, "pytorch_model.bin")

    target_device = torch.device(device)
    
    if os.path.exists(safetensors_path):
        # safetensors supports loading directly to device
        state_dict = load_file(safetensors_path, device=str(target_device))
    elif os.path.exists(pytorch_path):
        state_dict = torch.load(pytorch_path, map_location=target_device)
    else:
        raise ValueError(f"No model weights found in {load_directory}")

    # Convert dtype if needed
    if dtype is not None:
        state_dict = {k: v.to(dtype) if v.is_floating_point() else v 
                      for k, v in state_dict.items()}

    # Load state dict with assign=True for meta tensors
    if low_cpu_mem_usage:
        # First, move the model skeleton from meta to the target device
        # to_empty() creates empty tensors on the target device
        model = model.to_empty(device=target_device)
        
        # Convert dtype if needed before loading weights
        if dtype is not None:
            model = model.to(dtype)
        
        # Now load the weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
            
        # Re-initialize any modules that still have uninitialized tensors (like rotary embeddings)
        # These are computed buffers that need to be regenerated
        def fix_uninitialized_buffers(module, device, dtype_to_use):
            """Recursively fix any uninitialized buffers by reinitializing them."""
            for name, child in module.named_children():
                fix_uninitialized_buffers(child, device, dtype_to_use)

            # For rotary embeddings, recreate inv_freq and cos/sin caches
            if hasattr(module, 'inv_freq') and module.inv_freq is not None:
                buf = module.inv_freq
                # Check if buffer is uninitialized (all zeros or has NaN)
                if buf.numel() > 0 and (torch.isnan(buf).any() or (buf == 0).all()):
                    if hasattr(module, 'dim') and hasattr(module, 'base'):
                        dim = module.dim
                        base = module.base
                        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
                        module.register_buffer("inv_freq", inv_freq, persistent=False)

                # Always regenerate cos/sin caches if the module has them
                # These are non-persistent buffers that are never saved
                if hasattr(module, '_set_cos_sin_cache') and hasattr(module, 'max_position_embeddings'):
                    module._set_cos_sin_cache(module.max_position_embeddings, device=device)

        fix_uninitialized_buffers(model, target_device, dtype)
    else:
        model = model.to(target_device)
        if dtype is not None:
            model = model.to(dtype)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        # Filter out expected missing keys (e.g., rotary embeddings that are computed)
        real_missing = [k for k in missing_keys if "rotary" not in k.lower()]
        if real_missing:
            print(f"Warning: Missing keys in state dict: {real_missing[:5]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state dict: {unexpected_keys[:5]}...")

    # Ensure MLA compression module parameters are always float32
    # This is required for Riemannian optimization and numerical stability
    def ensure_compression_params_float32(module):
        """Ensure compression module parameters are float32."""
        import geoopt
        from .compression import MLACompression

        for name, child in module.named_children():
            ensure_compression_params_float32(child)

            # Check if this is a compression module
            if isinstance(child, MLACompression):
                # Convert all parameters in the compression module to float32
                if child.W_down_k.weight.dtype != torch.float32:
                    child.W_down_k.weight.data = child.W_down_k.weight.data.float()
                if child.W_down_v.weight.dtype != torch.float32:
                    child.W_down_v.weight.data = child.W_down_v.weight.data.float()
                if child.W_uk.dtype != torch.float32:
                    child.W_uk.data = child.W_uk.data.float()
                if child.W_uv.dtype != torch.float32:
                    child.W_uv.data = child.W_uv.data.float()

            # Also handle improved compression modules
            if hasattr(child, 'W_down') and hasattr(child, 'W_uk') and hasattr(child, 'W_uv'):
                if child.W_down.weight.dtype != torch.float32:
                    child.W_down.weight.data = child.W_down.weight.data.float()
                if child.W_uk.dtype != torch.float32:
                    child.W_uk.data = child.W_uk.data.float()
                if child.W_uv.dtype != torch.float32:
                    child.W_uv.data = child.W_uv.data.float()

        # Also handle any remaining ManifoldParameter instances
        for name, param in module.named_parameters(recurse=False):
            if isinstance(param, geoopt.ManifoldParameter):
                if param.dtype != torch.float32:
                    param.data = param.data.float()

    ensure_compression_params_float32(model)

    # Check for NaN/Inf in loaded weights
    def check_for_nan_inf(module, prefix=""):
        """Check for NaN/Inf values in model parameters."""
        issues = []
        for name, param in module.named_parameters():
            if torch.isnan(param).any():
                issues.append(f"{prefix}{name}: contains NaN values")
            if torch.isinf(param).any():
                issues.append(f"{prefix}{name}: contains Inf values")
        return issues

    issues = check_for_nan_inf(model)
    if issues:
        print(f"Warning: Found numerical issues in loaded model:")
        for issue in issues[:10]:  # Show first 10
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

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
