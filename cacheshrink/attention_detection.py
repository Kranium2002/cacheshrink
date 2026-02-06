"""Automatic attention type detection for selecting optimal compression method.

This module detects whether a model uses MHA, GQA, MQA, or native MLA attention
and recommends the appropriate compression method.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from transformers import PretrainedConfig


class AttentionType(Enum):
    """Enumeration of attention types supported by cacheshrink."""

    MHA = "mha"  # Multi-Head Attention (GPT-2, GPT-J, Pythia, LLaMA 2)
    GQA = "gqa"  # Grouped Query Attention (Qwen, Mistral, LLaMA 3)
    MQA = "mqa"  # Multi-Query Attention (Falcon, single KV head)
    NATIVE_MLA = "native_mla"  # Already compressed (DeepSeek V2, Kimi)
    UNKNOWN = "unknown"


@dataclass
class AttentionInfo:
    """Information about detected attention type and recommended compression method.

    Attributes:
        attention_type: Detected attention type (MHA, GQA, MQA, NATIVE_MLA, or UNKNOWN)
        n_heads: Number of query attention heads
        n_kv_heads: Number of key/value heads
        recommended_method: Recommended compression method ("mla", "xkv", or "unsupported")
        reason: Explanation for the recommendation
    """

    attention_type: AttentionType
    n_heads: int
    n_kv_heads: int
    recommended_method: str  # "mla", "xkv", or "unsupported"
    reason: str


def detect_attention_type(config: PretrainedConfig) -> AttentionInfo:
    """Detect attention type from HuggingFace model config.

    Analyzes the model configuration to determine:
    - MHA: n_kv_heads == n_heads (full redundancy, use per-layer MLA)
    - GQA: 1 < n_kv_heads < n_heads (limited per-layer redundancy, use xKV)
    - MQA: n_kv_heads == 1 (already maximally compressed per-layer, use xKV)
    - Native MLA: model has kv_lora_rank or q_lora_rank (already compressed)

    Args:
        config: HuggingFace PretrainedConfig object

    Returns:
        AttentionInfo with type, dimensions, and recommended compression method
    """
    # Extract attention head counts
    n_heads = getattr(config, "num_attention_heads", None)

    # Handle various naming conventions for n_kv_heads
    n_kv_heads = getattr(config, "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = getattr(config, "num_kv_heads", None)
    if n_kv_heads is None:
        n_kv_heads = getattr(config, "kv_heads", None)
    if n_kv_heads is None:
        # GPT-2 style: n_head attribute
        if hasattr(config, "n_head"):
            n_heads = config.n_head
            n_kv_heads = n_heads  # GPT-2 uses MHA
        else:
            n_kv_heads = n_heads  # Default to MHA if not specified

    # Handle None values
    if n_heads is None:
        return AttentionInfo(
            attention_type=AttentionType.UNKNOWN,
            n_heads=0,
            n_kv_heads=0,
            recommended_method="unsupported",
            reason="Could not determine number of attention heads from config",
        )

    if n_kv_heads is None:
        n_kv_heads = n_heads

    # Check for native MLA (DeepSeek V2, Kimi)
    has_kv_lora = hasattr(config, "kv_lora_rank") and config.kv_lora_rank is not None
    has_q_lora = hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
    if has_kv_lora or has_q_lora:
        return AttentionInfo(
            attention_type=AttentionType.NATIVE_MLA,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            recommended_method="unsupported",
            reason="Model already uses native MLA compression (DeepSeek V2 / Kimi style)",
        )

    # Detect based on head counts
    if n_kv_heads == n_heads:
        return AttentionInfo(
            attention_type=AttentionType.MHA,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            recommended_method="separate",
            reason="MHA has full per-head redundancy - use per-layer separate compression",
        )
    elif n_kv_heads == 1:
        return AttentionInfo(
            attention_type=AttentionType.MQA,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            recommended_method="xkv",
            reason="MQA already maximally compressed per-layer - use cross-layer xKV",
        )
    elif n_kv_heads < n_heads:
        return AttentionInfo(
            attention_type=AttentionType.GQA,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            recommended_method="xkv",
            reason="GQA has limited per-layer redundancy - use cross-layer xKV",
        )
    else:
        return AttentionInfo(
            attention_type=AttentionType.UNKNOWN,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            recommended_method="unsupported",
            reason=f"Unknown attention pattern: {n_heads} heads, {n_kv_heads} KV heads",
        )


def get_compression_method(
    config: PretrainedConfig,
    user_method: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Get appropriate compression method, with optional user override.

    This function detects the model's attention type and returns the optimal
    compression method. Users can override with a specific method, in which
    case a warning is printed if it's suboptimal.

    Args:
        config: HuggingFace model config
        user_method: User-specified method ("separate", "xkv", "auto", or None)
            - "separate": Force per-layer separate compression
            - "xkv": Force cross-layer xKV compression
            - "auto": Auto-detect and use optimal method
            - None: Same as "auto"
        verbose: Print detection info

    Returns:
        Compression method to use ("separate" or "xkv")

    Raises:
        ValueError: If model is unsupported (native MLA) or user_method is incompatible
    """
    info = detect_attention_type(config)

    if verbose:
        print(f"Detected attention type: {info.attention_type.value.upper()}")
        print(f"  Heads: {info.n_heads}, KV heads: {info.n_kv_heads}")
        print(f"  Recommended method: {info.recommended_method}")
        print(f"  Reason: {info.reason}")

    # Check for unsupported
    if info.recommended_method == "unsupported":
        raise ValueError(
            f"Model not supported for compression: {info.reason}\n"
            f"Detected attention type: {info.attention_type.value}"
        )

    # Handle user override
    if user_method is None or user_method == "auto":
        return info.recommended_method

    # Validate user method
    valid_methods = ["mla", "xkv", "separate", "joint", "decoupled_rope"]
    if user_method not in valid_methods:
        raise ValueError(
            f"Unknown compression method: {user_method}. "
            f"Valid options: {valid_methods}"
        )

    # Map legacy per-layer methods to "mla"
    if user_method in ["separate", "joint", "decoupled_rope"]:
        effective_method = "mla"
    else:
        effective_method = user_method

    # Warn if user chooses suboptimal method
    if effective_method == "mla" and info.attention_type in [
        AttentionType.GQA,
        AttentionType.MQA,
    ]:
        print(f"WARNING: Per-layer MLA not recommended for {info.attention_type.value}.")
        print(f"  {info.reason}")
        print("  Consider using compression_method='xkv' or 'auto' instead.")

    if effective_method == "xkv" and info.attention_type == AttentionType.MHA:
        print(f"NOTE: 'xkv' works but 'mla' is more efficient for MHA models.")

    return user_method


def is_gqa_model(config: PretrainedConfig) -> bool:
    """Check if model uses Grouped Query Attention.

    Args:
        config: HuggingFace model config

    Returns:
        True if model uses GQA (n_kv_heads < n_heads and n_kv_heads > 1)
    """
    info = detect_attention_type(config)
    return info.attention_type == AttentionType.GQA


def is_mqa_model(config: PretrainedConfig) -> bool:
    """Check if model uses Multi-Query Attention.

    Args:
        config: HuggingFace model config

    Returns:
        True if model uses MQA (n_kv_heads == 1)
    """
    info = detect_attention_type(config)
    return info.attention_type == AttentionType.MQA


def is_mha_model(config: PretrainedConfig) -> bool:
    """Check if model uses standard Multi-Head Attention.

    Args:
        config: HuggingFace model config

    Returns:
        True if model uses MHA (n_kv_heads == n_heads)
    """
    info = detect_attention_type(config)
    return info.attention_type == AttentionType.MHA
