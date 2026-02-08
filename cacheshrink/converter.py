"""Main conversion function for transforming HuggingFace models to MLA."""

from typing import Tuple, Optional, List, Union, Literal, Dict
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
CompressionMethod = Literal["separate", "joint", "decoupled_rope", "xkv", "auto"]


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
    calibration_dataset_subset: str = "wikitext-2-raw-v1",
    num_calibration_samples: int = 128,
    max_calibration_length: int = 512,
    trust_remote_code: bool = True,
    use_randomized_svd: bool = False,
    store_original_weights: bool = False,
    verbose: bool = True,
    # New xKV parameters
    auto_detect: bool = False,
    cross_layer_group_size: int = 4,
    xkv_skip_early_layers: int = 0,
    keep_early_layers_original: bool = False,
) -> Tuple[nn.Module, "AutoTokenizer"]:
    """Convert a HuggingFace model to use Multi-Head Latent Attention.

    This function:
    1. Loads the model and tokenizer
    2. Extracts configuration
    3. Optionally collects calibration data
    4. Converts each attention layer to MLAAttention
    5. Initializes compression matrices using SVD

    BACKWARD COMPATIBILITY:
    - Default behavior is UNCHANGED (compression_method="separate")
    - Existing scripts like benchmark_qwen.py work without modification
    - Auto-detection is OPT-IN via auto_detect=True or compression_method="auto"

    Args:
        model_name_or_path: HuggingFace model name or path
        compression_ratio: Target KV cache compression ratio (4-16x typical)
        compression_method: Compression method to use:
            EXISTING (unchanged):
            - "separate": Per-layer separate K/V compression (default, best quality)
            - "joint": Per-layer joint K/V with shared latent (DeepSeek-style)
            - "decoupled_rope": Per-layer with RoPE preservation
            NEW:
            - "xkv": Cross-layer xKV compression (recommended for GQA models)
            - "auto": Auto-detect and use optimal method
        d_latent: Override latent dimension (auto-computed if None)
        d_rope: RoPE dimension for decoupled_rope method (default 64)
        device: Device for the model ("cuda", "cpu", etc.)
        dtype: Data type for the model (torch.float16, etc.)
        use_calibration: Whether to use calibration data for initialization
        calibration_texts: Custom texts for calibration (loads dataset if None)
        calibration_dataset: HuggingFace dataset name for calibration
        calibration_dataset_subset: Dataset subset name (e.g., "wikitext-2-raw-v1")
        num_calibration_samples: Number of calibration samples
        max_calibration_length: Maximum sequence length for calibration
        trust_remote_code: Whether to trust remote code (needed for some models)
        use_randomized_svd: Use fast randomized SVD (50x faster, <1% accuracy loss)
        store_original_weights: Store original W_k/W_v as frozen buffers for reconstruction
            loss during training. Required for training with use_reconstruction_loss=True.
        verbose: Whether to print progress
        auto_detect: If True, auto-detect attention type and recommend optimal method.
            Prints warning if current method is suboptimal. Default: False
        cross_layer_group_size: Number of layers per xKV compression group (default: 4)
        xkv_skip_early_layers: Number of early layers to skip from xKV compression.
            These layers use per-layer MLA or original attention (see keep_early_layers_original).
            Default: 0
        keep_early_layers_original: If True, early layers (below xkv_skip_early_layers) are
            kept as original attention without any compression. If False, they use per-layer
            MLA compression. Default: False

    Returns:
        Tuple of (model, tokenizer) where model has MLA attention

    Raises:
        ValueError: If auto_detect=True and model uses native MLA (already compressed)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    valid_methods = ("separate", "joint", "decoupled_rope", "xkv", "auto")
    if compression_method not in valid_methods:
        raise ValueError(f"Unknown compression_method: {compression_method}. "
                        f"Must be one of {valid_methods}")

    # Handle auto-detection
    effective_method = compression_method
    use_xkv = False

    if auto_detect or compression_method == "auto":
        from .attention_detection import detect_attention_type, AttentionType

        hf_config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        info = detect_attention_type(hf_config)

        if verbose:
            print(f"Detected attention type: {info.attention_type.value.upper()}")
            print(f"  Heads: {info.n_heads}, KV heads: {info.n_kv_heads}")
            print(f"  Recommended: {info.recommended_method}")

        if info.recommended_method == "unsupported":
            raise ValueError(
                f"Model not supported for compression: {info.reason}\n"
                f"Detected attention type: {info.attention_type.value}"
            )

        if compression_method == "auto":
            effective_method = info.recommended_method
            if verbose:
                print(f"  Using auto-selected method: {effective_method}")
        else:
            # User specified method but also wants detection - warn if suboptimal
            if compression_method in ["separate", "joint", "decoupled_rope"]:
                if info.attention_type in [AttentionType.GQA, AttentionType.MQA]:
                    print(f"WARNING: Per-layer MLA not recommended for {info.attention_type.value}")
                    print(f"  {info.reason}")
                    print(f"  Consider using compression_method='xkv' or 'auto' instead.")

    if effective_method == "xkv":
        use_xkv = True
        # For xKV, use "separate" style MLAAttention but with cross-layer compression
        effective_method = "separate"

    # Validate xKV-only parameters aren't used with non-xKV methods
    if not use_xkv and (xkv_skip_early_layers > 0 or keep_early_layers_original):
        raise ValueError(
            "xkv_skip_early_layers and keep_early_layers_original are only supported "
            "with xKV compression (compression_method='xkv' or 'auto' with a GQA/MQA model). "
            f"Current method: '{compression_method}'"
        )

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load model and tokenizer
    if verbose:
        print(f"Loading model: {model_name_or_path}")
        print(f"Compression method: {compression_method}" + (" (xKV cross-layer)" if use_xkv else ""))

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
    
    # Store the *resolved* compression method (not "auto") so save/load round-trips work
    mla_config.compression_method = effective_method
    mla_config.d_rope = d_rope
    mla_config.use_cross_layer = use_xkv
    mla_config.cross_layer_group_size = cross_layer_group_size
    mla_config.xkv_skip_early_layers = xkv_skip_early_layers
    mla_config.keep_early_layers_original = keep_early_layers_original

    if verbose:
        print(f"MLA Config: {mla_config}")
        if use_xkv:
            print(f"xKV: {mla_config.n_groups} groups of {cross_layer_group_size} layers")
            if xkv_skip_early_layers > 0:
                if keep_early_layers_original:
                    print(f"     Keeping layers 0-{xkv_skip_early_layers-1} as original (no compression)")
                else:
                    print(f"     Skipping first {xkv_skip_early_layers} layers (using per-layer MLA)")

    # Get model handler
    handler = get_handler(model, mla_config)

    # Auto-detect bias from actual model weights (overrides config heuristic)
    b_q, b_k, b_v, b_o = handler.extract_qkv_biases(0)
    detected_bias = any(b is not None for b in (b_q, b_k, b_v, b_o))
    if detected_bias != mla_config.use_bias:
        if verbose:
            print(f"Auto-detected use_bias={detected_bias} from model weights "
                  f"(config heuristic was {mla_config.use_bias})")
        mla_config.use_bias = detected_bias

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
            dataset_config=calibration_dataset_subset,
            num_samples=num_calibration_samples,
            max_length=max_calibration_length,
            device=device,
            handler=handler,
        )
        if verbose:
            print(f"Collected calibration data for {len(calibration_data)} layers")

    # Get attention adapter class
    adapter_class = get_attention_adapter(mla_config.model_type)

    # xKV compression groups (initialized once, shared across layers)
    xkv_groups: Optional[Dict[int, "XKVCompressionGroup"]] = None

    if use_xkv:
        # Initialize xKV compression groups using cross-layer SVD
        from .xkv_initialization import cross_layer_svd_init

        if verbose:
            print("Initializing xKV compression groups with cross-layer SVD...")

        xkv_groups = cross_layer_svd_init(
            handler=handler,
            config=mla_config,
            calibration_data=calibration_data,
            verbose=verbose,
        )

        # Store groups on model as nn.ModuleDict for proper state_dict handling
        # This ensures shared W_uk/W_uv references are preserved during save/load
        model.xkv_groups = nn.ModuleDict({str(k): v for k, v in xkv_groups.items()})

    # Convert each layer
    n_layers = handler.get_num_layers()
    iterator = range(n_layers)
    if verbose:
        iterator = tqdm(iterator, desc="Converting layers")

    for layer_idx in iterator:
        # Check if this is an early layer that should be kept as original
        is_early_layer = use_xkv and not mla_config.is_xkv_layer(layer_idx)

        if is_early_layer and keep_early_layers_original:
            # Keep this layer as original attention - do not convert
            continue

        # Extract original weights
        W_q, W_k, W_v, W_o = handler.extract_qkv_weights(layer_idx)
        b_q, b_k, b_v, b_o = handler.extract_qkv_biases(layer_idx)

        # Get calibration data for this layer
        layer_calibration = None
        if calibration_data is not None and layer_idx in calibration_data:
            layer_calibration = calibration_data[layer_idx].to(device, dtype=W_k.dtype)

        # Create MLA attention module with appropriate compression
        mla_attn = MLAAttention(mla_config, layer_idx=layer_idx,
                                compression_method=effective_method, d_rope=d_rope)

        # Initialize compression based on method
        if use_xkv and mla_config.is_xkv_layer(layer_idx):
            # Use pre-initialized xKV compression module for non-early layers
            group_idx = mla_config.get_layer_group(layer_idx)
            xkv_compression = model.xkv_groups[str(group_idx)].get_compression(layer_idx)
            mla_attn.mla_compression = xkv_compression
            # Original weights already stored during xKV init
        elif use_xkv and not mla_config.is_xkv_layer(layer_idx):
            # Early layer: use per-layer MLA instead of xKV (only if keep_early_layers_original=False)
            _init_separate_compression(
                mla_attn, W_k, W_v, mla_config.computed_d_latent,
                layer_calibration, use_randomized_svd
            )
            if store_original_weights:
                mla_attn.mla_compression.store_original_weights(W_k, W_v)
        elif effective_method == "separate":
            _init_separate_compression(
                mla_attn, W_k, W_v, mla_config.computed_d_latent,
                layer_calibration, use_randomized_svd
            )
            # Store original weights for reconstruction loss if requested
            if store_original_weights:
                mla_attn.mla_compression.store_original_weights(W_k, W_v)
        elif effective_method == "joint":
            _init_joint_compression(
                mla_attn, W_k, W_v, mla_config.computed_d_latent,
                layer_calibration
            )
            if store_original_weights:
                mla_attn.mla_compression.store_original_weights(W_k, W_v)
        elif effective_method == "decoupled_rope":
            _init_decoupled_rope_compression(
                mla_attn, W_k, W_v, mla_config.computed_d_latent,
                d_rope, layer_calibration
            )
            if store_original_weights:
                mla_attn.mla_compression.store_original_weights(W_k, W_v)

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
        from .model_handlers.gpt2 import GPT2AttentionAdapter
        if adapter_class is GPT2AttentionAdapter:
            adapted_attn = adapter_class(mla_attn)
        else:
            adapted_attn = adapter_class(mla_attn, mla_config)

        # Replace attention in model
        handler.replace_attention(layer_idx, adapted_attn)

    # Attach MLA config and handler to model for later use
    model.mla_config = mla_config
    model.mla_handler = handler

    # Clear CUDA cache if using GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if verbose:
        print("Conversion complete!")
        _print_compression_stats(model, mla_config, compression_method, use_xkv, handler=handler)

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


def _print_compression_stats(
    model: nn.Module, config: MLAConfig, method: str, use_xkv: bool = False,
    handler=None,
) -> None:
    """Print compression statistics."""
    # Calculate cache size based on method
    if use_xkv or method == "xkv":
        compressed_dim = 2 * config.computed_d_latent
        method_desc = f"xKV Cross-Layer ({config.n_groups} groups)"
    elif method == "separate":
        compressed_dim = 2 * config.computed_d_latent
        method_desc = "Separate K/V"
    elif method == "joint":
        compressed_dim = config.computed_d_latent
        method_desc = "Joint K/V (DeepSeek-style)"
    elif method == "decoupled_rope":
        d_rope = getattr(config, "d_rope", 64)
        compressed_dim = 2 * d_rope + 2 * config.computed_d_latent
        method_desc = "Decoupled RoPE"
    else:
        compressed_dim = 2 * config.computed_d_latent
        method_desc = method

    original_kv_dim = 2 * config.d_kv
    per_layer_compression_ratio = original_kv_dim / compressed_dim

    print(f"\nCompression Statistics ({method_desc}):")
    print(f"  Original KV dimension: {original_kv_dim}")
    print(f"  Compressed dimension: {compressed_dim}")
    print(f"  Per-layer compression ratio: {per_layer_compression_ratio:.2f}x")

    if use_xkv:
        print(f"  Cross-layer groups: {config.n_groups}")
        print(f"  Layers per group: {config.cross_layer_group_size}")

        # Calculate effective compression ratio considering skipped early layers
        keep_original = getattr(config, "keep_early_layers_original", True)
        skip_layers = getattr(config, "xkv_skip_early_layers", 0)

        if keep_original and skip_layers > 0:
            n_original_layers = skip_layers
            n_compressed_layers = config.n_layers - skip_layers

            # Total cache size: original layers use full KV, compressed use d_latent
            total_original_kv = config.n_layers * original_kv_dim
            total_compressed = (n_original_layers * original_kv_dim +
                               n_compressed_layers * compressed_dim)
            effective_ratio = total_original_kv / total_compressed

            print(f"  Original layers (no compression): {n_original_layers}")
            print(f"  Compressed layers (xKV): {n_compressed_layers}")
            print(f"  Effective compression ratio: {effective_ratio:.2f}x")

    # Check orthonormality
    max_errors = []
    for layer_idx in range(config.n_layers):
        if handler is not None:
            layer = handler.get_layer_module(layer_idx)
            attn = getattr(layer, handler.get_attention_attribute_name())
        elif config.model_type == "gpt2":
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
