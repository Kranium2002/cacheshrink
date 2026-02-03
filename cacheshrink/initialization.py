"""Initialization methods for MLA compression matrices."""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any
from tqdm import tqdm
import warnings

from .config import MLAConfig
from .utils import orthonormalize_columns
from .fast_svd import randomized_svd


def compute_optimal_d_latent(
    W: torch.Tensor,
    energy_threshold: float = 0.95,
) -> Tuple[int, float]:
    """Compute optimal d_latent to capture given energy threshold.

    Args:
        W: Weight matrix of shape (d_kv, d_model)
        energy_threshold: Fraction of energy to capture (0.0-1.0)

    Returns:
        Tuple of (optimal_d_latent, actual_energy_captured)
    """
    W_f32 = W.float()
    _, S, _ = torch.linalg.svd(W_f32, full_matrices=False)

    total_energy = (S ** 2).sum()
    cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy

    # Find smallest d_latent that captures threshold energy
    d_latent = int((cumulative_energy >= energy_threshold).float().argmax().item()) + 1
    actual_energy = cumulative_energy[d_latent - 1].item()

    return d_latent, actual_energy


def analyze_compression_quality(
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    d_latent: int,
) -> Dict[str, float]:
    """Analyze compression quality for given d_latent.

    Args:
        W_k: Key projection weights of shape (d_kv, d_model)
        W_v: Value projection weights of shape (d_kv, d_model)
        d_latent: Target latent dimension

    Returns:
        Dictionary with energy captured and reconstruction errors for K and V
    """
    W_k_f32 = W_k.float()
    W_v_f32 = W_v.float()

    _, S_k, _ = torch.linalg.svd(W_k_f32, full_matrices=False)
    _, S_v, _ = torch.linalg.svd(W_v_f32, full_matrices=False)

    energy_k = (S_k[:d_latent] ** 2).sum() / (S_k ** 2).sum()
    energy_v = (S_v[:d_latent] ** 2).sum() / (S_v ** 2).sum()

    return {
        "k_energy": energy_k.item(),
        "v_energy": energy_v.item(),
        "k_recon_error": 1 - energy_k.item(),  # Approximate
        "v_recon_error": 1 - energy_v.item(),  # Approximate
    }


def balanced_svd_init(
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    d_latent: int,
    calibration_data: Optional[torch.Tensor] = None,
    use_randomized_svd: bool = True,
    svd_oversamples: int = 20,
    svd_power_iterations: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize MLA compression matrices using SVD.

    Two modes:
    1. Without calibration: SVD on weight matrices W_k, W_v directly
    2. With calibration: SVD on actual K=H@W_k.T, V=H@W_v.T from calibration data
       This finds the optimal subspace for the actual data distribution.

    Args:
        W_k: Key projection weights of shape (d_kv, d_model)
        W_v: Value projection weights of shape (d_kv, d_model)
        d_latent: Target latent dimension
        calibration_data: Optional hidden states of shape (n_tokens, d_model)
            If provided, uses calibration-aware SVD which typically gives much
            better compression quality.
        use_randomized_svd: Use fast randomized SVD
        svd_oversamples: Oversampling parameter for randomized SVD
        svd_power_iterations: Power iterations for randomized SVD

    Returns:
        Tuple of (W_down_k, W_down_v, W_uk, W_uv) where:
            - W_down_k: (d_latent, d_model) K compression matrix
            - W_down_v: (d_latent, d_model) V compression matrix
            - W_uk: (d_kv, d_latent) K decompression (orthonormal columns)
            - W_uv: (d_kv, d_latent) V decompression (orthonormal columns)
    """
    d_kv, d_model = W_k.shape
    device = W_k.device
    dtype = W_k.dtype

    # Check compression quality and warn if too aggressive
    quality = analyze_compression_quality(W_k, W_v, d_latent)
    if quality["v_energy"] < 0.90:
        warnings.warn(
            f"Aggressive compression detected! Value projection only captures {quality['v_energy']*100:.1f}% "
            f"of energy with d_latent={d_latent}. This will likely cause poor reconstruction. "
            f"Consider using a smaller compression_ratio (larger d_latent). "
            f"Recommended: d_latent >= {compute_optimal_d_latent(W_v, 0.90)[0]} for 90% energy."
        )
    elif quality["v_energy"] < 0.95:
        warnings.warn(
            f"Value projection captures only {quality['v_energy']*100:.1f}% of energy with d_latent={d_latent}. "
            f"Consider using compression_ratio=2.0 for better quality."
        )

    # SVD requires float32
    compute_dtype = torch.float32 if dtype == torch.float16 or dtype == torch.bfloat16 else dtype
    W_k_f32 = W_k.to(compute_dtype)
    W_v_f32 = W_v.to(compute_dtype)

    # Use calibration-aware SVD if calibration data is provided
    if calibration_data is not None and calibration_data.numel() > 0:
        # Calibration-aware SVD: find optimal subspace from actual K/V outputs
        from .improved_compression import calibration_aware_svd

        W_down_k, W_down_v, W_uk, W_uv, stats = calibration_aware_svd(
            W_k, W_v, calibration_data, d_latent,
            max_samples=min(10000, calibration_data.size(0))
        )

        # Log improvement (calibration typically captures 10-20% more energy)
        if stats["v_energy"] > quality["v_energy"] + 0.05:
            print(f"  Calibration-aware SVD improved V energy: {quality['v_energy']*100:.1f}% -> {stats['v_energy']*100:.1f}%")

        return W_down_k.to(dtype), W_down_v.to(dtype), W_uk.to(dtype), W_uv.to(dtype)

    # Fallback to weight-only SVD
    # SVD of W_k: W_k = U_k @ S_k @ Vh_k
    if use_randomized_svd:
        U_k, S_k, _ = randomized_svd(W_k_f32, d_latent, n_oversamples=svd_oversamples, n_power_iterations=svd_power_iterations)
    else:
        U_k_full, S_k_full, _ = torch.linalg.svd(W_k_f32, full_matrices=False)
        U_k = U_k_full[:, :d_latent]

    # SVD of W_v: W_v = U_v @ S_v @ Vh_v
    if use_randomized_svd:
        U_v, S_v, _ = randomized_svd(W_v_f32, d_latent, n_oversamples=svd_oversamples, n_power_iterations=svd_power_iterations)
    else:
        U_v_full, S_v_full, _ = torch.linalg.svd(W_v_f32, full_matrices=False)
        U_v = U_v_full[:, :d_latent]

    # U_k and U_v have orthonormal columns (SVD property)
    W_uk = U_k  # (d_kv, d_latent)
    W_uv = U_v  # (d_kv, d_latent)

    # Optimal W_down for each: W_down = W_u.T @ W
    # Since W_u has orthonormal columns: W_u @ W_u.T @ W = projection of W onto col(W_u)
    # And W_u @ (W_u.T @ W) reconstructs W optimally
    W_down_k = W_uk.T @ W_k_f32  # (d_latent, d_model)
    W_down_v = W_uv.T @ W_v_f32  # (d_latent, d_model)

    return W_down_k.to(dtype), W_down_v.to(dtype), W_uk.to(dtype), W_uv.to(dtype)


def collect_calibration_data(
    model: nn.Module,
    tokenizer,
    texts: Optional[List[str]] = None,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    num_samples: int = 128,
    max_length: int = 512,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> Dict[int, torch.Tensor]:
    """Collect calibration data (hidden states) from model.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: Optional list of texts (if None, loads from dataset)
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        split: Dataset split
        num_samples: Number of samples to collect
        max_length: Maximum sequence length
        batch_size: Batch size for inference
        device: Device for computation

    Returns:
        Dictionary mapping layer index to hidden states tensor
        Each tensor has shape (total_tokens, d_model)
    """
    from datasets import load_dataset

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Load texts if not provided
    if texts is None:
        dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
        texts = []
        text_key = "text" if "text" in dataset.column_names else dataset.column_names[0]
        for item in dataset:
            if item[text_key] and len(item[text_key].strip()) > 100:
                texts.append(item[text_key])
                if len(texts) >= num_samples:
                    break

    # Storage for hidden states per layer
    layer_hidden_states: Dict[int, List[torch.Tensor]] = {}
    n_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else model.config.n_layer

    for i in range(n_layers):
        layer_hidden_states[i] = []

    # Hook to capture hidden states
    hooks = []
    captured_states = {}

    def make_hook(layer_idx):
        def hook(module, args, kwargs):
            # Get hidden states from args or kwargs
            if args and len(args) > 0:
                hidden = args[0]
            elif "hidden_states" in kwargs:
                hidden = kwargs["hidden_states"]
            else:
                return
            captured_states[layer_idx] = hidden.detach()
        return hook

    # Register hooks on decoder layers (not attention) to capture hidden states
    for layer_idx in range(n_layers):
        # Find decoder layer module
        if hasattr(model, "transformer"):
            # GPT-2 style
            layer = model.transformer.h[layer_idx]
        elif hasattr(model, "model"):
            # LLaMA/Mistral/Qwen style
            layer = model.model.layers[layer_idx]
        else:
            raise ValueError("Unknown model architecture")

        hook = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
        hooks.append(hook)

    try:
        # Process texts in batches
        for batch_start in tqdm(range(0, len(texts), batch_size), desc="Collecting calibration data"):
            batch_texts = texts[batch_start:batch_start + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            # Forward pass
            with torch.no_grad():
                _ = model(**inputs)

            # Collect hidden states (only non-padding tokens)
            attention_mask = inputs.get("attention_mask", None)

            for layer_idx, hidden in captured_states.items():
                if attention_mask is not None:
                    # Flatten and select non-padding tokens
                    mask = attention_mask.bool().view(-1)
                    hidden_flat = hidden.view(-1, hidden.size(-1))
                    hidden_selected = hidden_flat[mask]
                else:
                    hidden_selected = hidden.view(-1, hidden.size(-1))

                layer_hidden_states[layer_idx].append(hidden_selected.cpu())

            captured_states.clear()

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    # Concatenate all collected states per layer
    result = {}
    for layer_idx, states in layer_hidden_states.items():
        if states:
            result[layer_idx] = torch.cat(states, dim=0)

    return result


def init_compression_from_calibration(
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    d_latent: int,
    hidden_states: torch.Tensor,
    max_calibration_samples: int = 10000,
    use_randomized_svd: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize compression matrices using calibration data.

    This is a convenience wrapper around balanced_svd_init that handles
    sampling from large calibration datasets.

    Args:
        W_k: Key projection weights of shape (d_kv, d_model)
        W_v: Value projection weights of shape (d_kv, d_model)
        d_latent: Target latent dimension
        hidden_states: Calibration hidden states of shape (n_tokens, d_model)
        max_calibration_samples: Maximum samples to use for balancing
        use_randomized_svd: Use fast randomized SVD (default True)

    Returns:
        Tuple of (W_down_k, W_down_v, W_uk, W_uv)
    """
    # Subsample if needed
    if hidden_states.size(0) > max_calibration_samples:
        indices = torch.randperm(hidden_states.size(0))[:max_calibration_samples]
        hidden_states = hidden_states[indices]

    # Move to same device as weights
    hidden_states = hidden_states.to(W_k.device, W_k.dtype)

    return balanced_svd_init(W_k, W_v, d_latent, hidden_states, use_randomized_svd=use_randomized_svd)
