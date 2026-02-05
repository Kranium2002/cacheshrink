"""Main conversion function for transforming HuggingFace models to MLA."""

from typing import Tuple, Optional, List, Union, Literal
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import MLAConfig
from .attention import MLAAttention
from .initialization import balanced_svd_init, collect_calibration_data, init_compression_from_calibration
from .model_handlers import get_handler, get_attention_adapter
from .improved_compression import (
    JointKVCompression,
    DecoupledRoPECompression,
    calibration_aware_svd,
    joint_kv_svd,
    init_decoupled_rope_from_weights,
)


# Type alias for compression methods
CompressionMethod = Literal["separate", "joint", "decoupled_rope"]


def convert_to_mla(
    model_name_or_path: str,
    compression_ratio: float = 4.0,
    compression_method: CompressionMethod = "separate",
    d_latent: Optional[int] = None,
    d_rope: int = 64,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    use_calibration: bool = True,
    calibration_texts: Optional[List[str]] = None,
    calibration_dataset: str = "wikitext",
    calibration_config: str = "wikitext-2-raw-v1",
    num_calibration_samples: int = 128,
    max_calibration_length: int = 512,
    trust_remote_code: bool = True,
    use_randomized_svd: bool = False,
    verbose: bool = True,
) -> Tuple[nn.Module, "AutoTokenizer"]:
    """Convert a HuggingFace model to use Multi-Head Latent Attention.

    This function:
    1. Loads the model and tokenizer
    2. Extracts configuration
    3. Optionally collects calibration data
    4. Converts each attention layer to MLAAttention
    5. Initializes compression matrices using SVD

    Args:
        model_name_or_path: HuggingFace model name or path
        compression_ratio: Target KV cache compression ratio (4-16x typical)
        compression_method: Compression method to use:
            - "separate": Separate K/V compression (default, best quality)
            - "joint": Joint K/V with shared latent (DeepSeek-style, 2x more compression)
            - "decoupled_rope": Decoupled RoPE (preserves positional info, best for long context)
        d_latent: Override latent dimension (auto-computed if None)
        d_rope: RoPE dimension for decoupled_rope method (default 64)
        device: Device for the model ("cuda", "cpu", etc.)
        dtype: Data type for the model (torch.float16, etc.)
        use_calibration: Whether to use calibration data for initialization
        calibration_texts: Custom texts for calibration (loads dataset if None)
        calibration_dataset: HuggingFace dataset name for calibration
        calibration_config: Dataset configuration
        num_calibration_samples: Number of calibration samples
        max_calibration_length: Maximum sequence length for calibration
        trust_remote_code: Whether to trust remote code (needed for some models)
        use_randomized_svd: Use fast randomized SVD (50x faster, <1% accuracy loss)
        verbose: Whether to print progress

    Returns:
        Tuple of (model, tokenizer) where model has MLA attention
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if compression_method not in ("separate", "joint", "decoupled_rope"):
        raise ValueError(f"Unknown compression_method: {compression_method}. "
                        f"Must be 'separate', 'joint', or 'decoupled_rope'")

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load model and tokenizer
    if verbose:
        print(f"Loading model: {model_name_or_path}")
        print(f"Compression method: {compression_method}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map=None,
    )
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create MLA config
    mla_config = MLAConfig.from_pretrained(
        model_name_or_path,
        compression_ratio=compression_ratio,
        d_latent=d_latent,
        trust_remote_code=trust_remote_code,
    )
    
    # Store compression method in config for later use
    mla_config.compression_method = compression_method
    mla_config.d_rope = d_rope

    if verbose:
        print(f"MLA Config: {mla_config}")

    # Get model handler
    handler = get_handler(model, mla_config)

    # Collect calibration data if requested
    calibration_data = None
    if use_calibration:
        if verbose:
            print("Collecting calibration data...")
        calibration_data = collect_calibration_data(
            model=model,
            tokenizer=tokenizer,
            texts=calibration_texts,
            dataset_name=calibration_dataset,
            dataset_config=calibration_config,
            num_samples=num_calibration_samples,
            max_length=max_calibration_length,
            device=device,
        )
        if verbose:
            print(f"Collected calibration data for {len(calibration_data)} layers")

    # Get attention adapter class
    adapter_class = get_attention_adapter(mla_config.model_type)

    # Convert each layer
    n_layers = handler.get_num_layers()
    iterator = range(n_layers)
    if verbose:
        iterator = tqdm(iterator, desc="Converting layers")

    for layer_idx in iterator:
        # Extract original weights
        W_q, W_k, W_v, W_o = handler.extract_qkv_weights(layer_idx)
        b_q, b_k, b_v, b_o = handler.extract_qkv_biases(layer_idx)

        # Get calibration data for this layer
        layer_calibration = None
        if calibration_data is not None and layer_idx in calibration_data:
            layer_calibration = calibration_data[layer_idx].to(device, dtype=W_k.dtype)

        # Create MLA attention module with appropriate compression
        mla_attn = MLAAttention(mla_config, layer_idx=layer_idx, 
                                compression_method=compression_method, d_rope=d_rope)

        # Initialize compression based on method
        if compression_method == "separate":
            _init_separate_compression(
                mla_attn, W_k, W_v, mla_config.computed_d_latent,
                layer_calibration, use_randomized_svd
            )
        elif compression_method == "joint":
            _init_joint_compression(
                mla_attn, W_k, W_v, mla_config.computed_d_latent,
                layer_calibration
            )
        elif compression_method == "decoupled_rope":
            _init_decoupled_rope_compression(
                mla_attn, W_k, W_v, mla_config.computed_d_latent,
                d_rope, layer_calibration
            )

        # Copy Q projection weights
        mla_attn.q_proj.weight.data = W_q.clone()
        if b_q is not None:
            mla_attn.q_proj.bias = nn.Parameter(b_q.clone())

        # Copy O projection weights
        mla_attn.o_proj.weight.data = W_o.clone()
        if b_o is not None:
            mla_attn.o_proj.bias = nn.Parameter(b_o.clone())

        # Move to device
        mla_attn = mla_attn.to(device)
        if dtype is not None:
            mla_attn = mla_attn.to(dtype)

        # Wrap with adapter for compatibility
        if mla_config.model_type == "gpt2":
            adapted_attn = adapter_class(mla_attn)
        else:
            adapted_attn = adapter_class(mla_attn, mla_config)

        # Replace attention in model
        handler.replace_attention(layer_idx, adapted_attn)

    # Attach MLA config to model for later use
    model.mla_config = mla_config

    # Clear CUDA cache if using GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if verbose:
        print("Conversion complete!")
        _print_compression_stats(model, mla_config, compression_method)

    return model, tokenizer


def _init_separate_compression(mla_attn, W_k, W_v, d_latent, calibration_data, use_randomized_svd):
    """Initialize separate K/V compression."""
    if calibration_data is not None:
        W_down_k, W_down_v, W_uk, W_uv = init_compression_from_calibration(
            W_k=W_k, W_v=W_v, d_latent=d_latent,
            hidden_states=calibration_data, use_randomized_svd=use_randomized_svd,
        )
    else:
        W_down_k, W_down_v, W_uk, W_uv = balanced_svd_init(
            W_k=W_k, W_v=W_v, d_latent=d_latent, use_randomized_svd=use_randomized_svd,
        )
    mla_attn.mla_compression.init_from_weights(W_down_k, W_down_v, W_uk, W_uv)


def _init_joint_compression(mla_attn, W_k, W_v, d_latent, calibration_data):
    """Initialize joint K/V compression."""
    if calibration_data is not None:
        W_down, W_uk, W_uv, _ = joint_kv_svd(
            W_k, W_v, calibration_data, d_latent
        )
    else:
        # Use random hidden states for initialization without calibration
        device = W_k.device
        H_random = torch.randn(1000, W_k.shape[1], device=device, dtype=W_k.dtype)
        W_down, W_uk, W_uv, _ = joint_kv_svd(W_k, W_v, H_random, d_latent)
    mla_attn.mla_compression.init_from_weights(W_down, W_uk, W_uv)


def _init_decoupled_rope_compression(mla_attn, W_k, W_v, d_latent, d_rope, calibration_data):
    """Initialize decoupled RoPE compression."""
    init_decoupled_rope_from_weights(
        mla_attn.mla_compression, W_k, W_v, calibration_data
    )


def _print_compression_stats(model: nn.Module, config: MLAConfig, method: str) -> None:
    """Print compression statistics."""
    # Calculate cache size based on method
    if method == "separate":
        compressed_dim = 2 * config.computed_d_latent
        method_desc = "Separate K/V"
    elif method == "joint":
        compressed_dim = config.computed_d_latent
        method_desc = "Joint K/V (DeepSeek-style)"
    elif method == "decoupled_rope":
        d_rope = getattr(config, 'd_rope', 64)
        compressed_dim = 2 * d_rope + 2 * config.computed_d_latent
        method_desc = "Decoupled RoPE"
    else:
        compressed_dim = 2 * config.computed_d_latent
        method_desc = method

    original_kv_dim = 2 * config.d_kv
    compression_ratio = original_kv_dim / compressed_dim

    print(f"\nCompression Statistics ({method_desc}):")
    print(f"  Original KV dimension: {original_kv_dim}")
    print(f"  Compressed dimension: {compressed_dim}")
    print(f"  Actual compression ratio: {compression_ratio:.2f}x")

    # Check orthonormality
    max_errors = []
    for layer_idx in range(config.n_layers):
        if config.model_type == "gpt2":
            attn = model.transformer.h[layer_idx].attn
        else:
            attn = model.model.layers[layer_idx].self_attn

        if hasattr(attn, "check_orthonormality"):
            errors = attn.check_orthonormality()
            max_errors.append(max(errors["W_uk"][0], errors["W_uv"][0]))
        elif hasattr(attn, "mla") and hasattr(attn.mla, "check_orthonormality"):
            errors = attn.mla.check_orthonormality()
            max_errors.append(max(errors["W_uk"][0], errors["W_uv"][0]))

    if max_errors:
        print(f"  Max orthonormality error: {max(max_errors):.2e}")
