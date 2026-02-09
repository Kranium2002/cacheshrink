"""Weight name mapping and loading for vLLM cacheshrink models.

Maps saved cacheshrink state dict keys to vLLM model parameter names.
Handles:
- Prefix remapping (model.model. -> model.)
- Attention adapter flattening (self_attn.mla. -> self_attn.)
- Compression module parameters (kept in float32)
- MLP gate/up merge for SwiGLU models
- xKV shared parameter loading

Saved state dict key format (from GenericAttentionAdapter):
    model.model.layers.{i}.self_attn.mla.q_proj.weight
    model.model.layers.{i}.self_attn.mla.mla_compression.W_down_k.weight
    model.model.layers.{i}.self_attn.mla.mla_compression.W_uk
    model.model.layers.{i}.self_attn.mla.o_proj.weight
    model.model.layers.{i}.mlp.*
    model.model.embed_tokens.weight
    lm_head.weight

vLLM model parameter names:
    model.layers.{i}.self_attn.q_proj.weight
    model.layers.{i}.self_attn.mla_compression.W_down_k.weight
    model.layers.{i}.self_attn.mla_compression.W_uk
    model.layers.{i}.self_attn.o_proj.weight
    model.layers.{i}.mlp.*
    model.embed_tokens.weight
    lm_head.weight
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from .config import VLLMMLAConfig

if TYPE_CHECKING:
    pass


# Keys that indicate compression parameters (must stay float32)
_COMPRESSION_KEYS = (
    "W_uk",
    "W_uv",
    "W_down_k.",
    "W_down_v.",
    "shared_W_uk",
    "shared_W_uv",
    "b_k",
    "b_v",
)

# Keys to skip during weight loading (not needed in vLLM model)
_SKIP_PATTERNS = (
    "W_k_original",
    "W_v_original",
    "rotary_emb.",
    "inv_freq",
    "cos_cached",
    "sin_cached",
)


def map_weight_name(saved_name: str, mla_config: VLLMMLAConfig) -> Optional[str]:
    """Map a saved state dict key to the corresponding vLLM model parameter name.

    Args:
        saved_name: Key from the saved state dict
        mla_config: MLA configuration

    Returns:
        Mapped parameter name for the vLLM model, or None if the key should be skipped
    """
    # Skip patterns
    for pattern in _SKIP_PATTERNS:
        if pattern in saved_name:
            return None

    name = saved_name

    # Remove the extra model. prefix: model.model.layers -> model.layers
    # Also handles: model.model.embed_tokens -> model.embed_tokens
    if name.startswith("model.model."):
        name = "model." + name[len("model.model.") :]

    # Handle transformer.h style (GPT-2)
    if name.startswith("model.transformer.h."):
        name = name.replace("model.transformer.h.", "model.layers.")

    # Flatten self_attn.mla. -> self_attn.
    # But keep mla_compression sub-path intact
    if ".self_attn.mla." in name:
        name = name.replace(".self_attn.mla.", ".self_attn.")

    # Handle xkv_groups keys (these live at model level)
    if name.startswith("xkv_groups."):
        # xkv_groups are loaded separately
        return None

    return name


def map_weight_names(state_dict_keys: Iterable[str], mla_config: VLLMMLAConfig) -> dict[str, str]:
    """Map all saved state dict keys to vLLM parameter names.

    Args:
        state_dict_keys: Keys from the saved state dict
        mla_config: MLA configuration

    Returns:
        Dict mapping saved_name -> vllm_name (excluding skipped keys)
    """
    mapping = {}
    for saved_name in state_dict_keys:
        vllm_name = map_weight_name(saved_name, mla_config)
        if vllm_name is not None:
            mapping[saved_name] = vllm_name
    return mapping


def is_compression_param(name: str) -> bool:
    """Check if a parameter name indicates a compression parameter (must stay float32)."""
    return any(ck in name for ck in _COMPRESSION_KEYS)


def load_cacheshrink_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    mla_config: VLLMMLAConfig,
) -> None:
    """Load weights into a CacheShrinkForCausalLM model.

    Handles name mapping, dtype management (compression params stay float32),
    and gate/up projection merging for SwiGLU MLPs.

    Args:
        model: The vLLM CacheShrinkForCausalLM model
        weights: Iterable of (name, tensor) pairs from the saved model
        mla_config: MLA configuration
    """
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    params_dict = dict(model.named_parameters())
    # Also include buffers for compression module biases
    for name, buf in model.named_buffers():
        if buf is not None:
            params_dict[name] = buf

    # Collect gate/up weights for merging (SwiGLU models)
    gate_up_buffer: dict[str, dict[str, torch.Tensor]] = {}

    for saved_name, tensor in weights:
        vllm_name = map_weight_name(saved_name, mla_config)
        if vllm_name is None:
            continue

        # Keep compression params in float32
        if is_compression_param(vllm_name):
            tensor = tensor.float()

        # Handle gate_proj / up_proj -> gate_up_proj merge for SwiGLU
        if ".gate_proj." in vllm_name or ".up_proj." in vllm_name:
            # Determine the merged name
            if ".gate_proj." in vllm_name:
                merged_name = vllm_name.replace(".gate_proj.", ".gate_up_proj.")
                proj_type = "gate"
            else:
                merged_name = vllm_name.replace(".up_proj.", ".gate_up_proj.")
                proj_type = "up"

            if merged_name not in gate_up_buffer:
                gate_up_buffer[merged_name] = {}
            gate_up_buffer[merged_name][proj_type] = tensor
            gate_up_buffer[merged_name][f"{proj_type}_saved_name"] = saved_name

            # Check if we have both parts
            if "gate" in gate_up_buffer[merged_name] and "up" in gate_up_buffer[merged_name]:
                gate_w = gate_up_buffer[merged_name]["gate"]
                up_w = gate_up_buffer[merged_name]["up"]
                merged_tensor = torch.cat([gate_w, up_w], dim=0)

                if merged_name in params_dict:
                    param = params_dict[merged_name]
                    default_weight_loader(param, merged_tensor)
                del gate_up_buffer[merged_name]
            continue

        # Standard weight loading
        if vllm_name in params_dict:
            param = params_dict[vllm_name]
            default_weight_loader(param, tensor)

    # Warn about any unmatched gate/up pairs
    for merged_name, parts in gate_up_buffer.items():
        loaded_parts = [k for k in parts if k in ("gate", "up")]
        if loaded_parts:
            print(
                f"Warning: Incomplete gate/up merge for {merged_name}, "
                f"only have: {loaded_parts}"
            )
