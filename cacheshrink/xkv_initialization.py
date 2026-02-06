"""Cross-layer SVD initialization for xKV compression.

This module provides functions to initialize xKV compression groups using
cross-layer SVD. The key insight is that adjacent layers have similar KV
structure, so we can find shared basis vectors across the group.

Initialization process:
1. Collect K/V from ALL layers in a group
2. Stack them: (group_size * n_tokens, d_kv)
3. SVD on stacked matrix to find shared basis
4. Per-layer projections map to this shared space
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import MLAConfig
from .xkv_compression import XKVCompressionGroup


def cross_layer_svd_init(
    handler,
    config: MLAConfig,
    calibration_data: Optional[Dict[int, torch.Tensor]] = None,
    verbose: bool = True,
) -> Dict[int, XKVCompressionGroup]:
    """Initialize xKV compression groups using cross-layer SVD.

    This is the core xKV initialization algorithm:
    1. For each group of layers:
       a. Extract original K/V weights for all layers
       b. If calibration data is available:
          - Compute K/V activations for each layer
          - Stack across layers
          - SVD on stacked matrix to find shared basis
       c. Otherwise:
          - Stack K/V weight matrices across layers
          - SVD on stacked weights
    2. Initialize shared W_uk, W_uv from top singular vectors
    3. Compute per-layer W_down as optimal projection to shared space

    Args:
        handler: ModelHandler for extracting weights from the model
        config: MLAConfig with cross-layer settings
        calibration_data: Dictionary mapping layer_idx to hidden states
            of shape (n_tokens, d_model). If None, uses weight-based init.
        verbose: Print initialization info

    Returns:
        Dictionary mapping group_idx to XKVCompressionGroup
    """
    groups = {}
    d_latent = config.computed_d_latent
    device = None

    # Determine device from calibration data or handler
    if calibration_data:
        first_key = next(iter(calibration_data.keys()))
        device = calibration_data[first_key].device

    if verbose:
        print(f"Initializing xKV compression with cross-layer SVD")
        print(f"  d_latent: {d_latent}")
        print(f"  group_size: {config.cross_layer_group_size}")
        print(f"  n_groups: {config.n_groups}")
        if config.xkv_skip_early_layers > 0:
            print(f"  Skipping layers 0-{config.xkv_skip_early_layers - 1} (per-layer MLA)")
            print(f"  xKV layers: {config.xkv_skip_early_layers}-{config.n_layers - 1}")

    for group_idx in range(config.n_groups):
        layer_indices = config.get_group_layers(group_idx)

        if verbose:
            print(f"\nGroup {group_idx}: layers {layer_indices}")

        # Collect weights, biases, and optionally activations from all layers in group
        W_k_layers = []
        W_v_layers = []
        b_k_layers = []
        b_v_layers = []
        K_stacked = []
        V_stacked = []

        for layer_idx in layer_indices:
            W_q, W_k, W_v, W_o = handler.extract_qkv_weights(layer_idx)
            b_q, b_k, b_v, b_o = handler.extract_qkv_biases(layer_idx)
            W_k_layers.append(W_k)
            W_v_layers.append(W_v)
            b_k_layers.append(b_k)
            b_v_layers.append(b_v)

            if device is None:
                device = W_k.device

            # Compute K/V activations if calibration data is available
            # Include biases in activation computation if present
            if calibration_data is not None and layer_idx in calibration_data:
                H = calibration_data[layer_idx].to(W_k.device).float()  # (n_tokens, d_model)
                K = H @ W_k.float().T  # (n_tokens, d_kv)
                V = H @ W_v.float().T
                # Add biases to get true K/V activations
                if b_k is not None:
                    K = K + b_k.float().to(K.device)
                if b_v is not None:
                    V = V + b_v.float().to(V.device)
                K_stacked.append(K)
                V_stacked.append(V)

        # Compute shared basis using SVD
        if K_stacked:
            # Use calibration-based initialization (recommended)
            shared_W_uk, shared_W_uv, k_energy, v_energy = _compute_shared_basis_from_activations(
                K_stacked, V_stacked, d_latent
            )
        else:
            # Fall back to weight-based initialization
            shared_W_uk, shared_W_uv, k_energy, v_energy = _compute_shared_basis_from_weights(
                W_k_layers, W_v_layers, d_latent
            )

        if verbose:
            print(f"  K energy retained: {k_energy:.1%}")
            print(f"  V energy retained: {v_energy:.1%}")

        # Create compression group with shared basis
        group = XKVCompressionGroup(config, layer_indices, d_latent)
        group.init_shared_basis(shared_W_uk.to(device), shared_W_uv.to(device))

        # IMPORTANT: Use the group's shared matrices (after orthonormalization)
        # to compute W_down, ensuring consistency between compression and decompression
        final_W_uk = group.shared_W_uk.data  # Use the actual stored matrix
        final_W_uv = group.shared_W_uv.data

        # Initialize per-layer W_down projections and store biases
        for i, layer_idx in enumerate(layer_indices):
            W_k = W_k_layers[i]
            W_v = W_v_layers[i]
            b_k = b_k_layers[i]
            b_v = b_v_layers[i]

            # Optimal W_down: minimize ||K - H @ W_down.T @ W_u.T||
            # Solution: W_down = W_u.T @ W_original
            # This gives W_down of shape (d_latent, d_model)
            W_down_k = final_W_uk.T @ W_k.float().to(final_W_uk.device)  # (d_latent, d_model)
            W_down_v = final_W_uv.T @ W_v.float().to(final_W_uv.device)

            comp = group.get_compression(layer_idx)
            comp.W_down_k.weight.data = W_down_k.to(device)
            comp.W_down_v.weight.data = W_down_v.to(device)
            comp.store_original_weights(W_k, W_v)
            # Store biases (critical for models like Qwen with non-zero K/V biases)
            comp.store_biases(b_k, b_v)

        # Verify reconstruction quality (sanity check)
        if verbose and K_stacked:
            # Test reconstruction on first layer of group
            first_layer = layer_indices[0]
            test_comp = group.get_compression(first_layer)
            W_k_test = W_k_layers[0].to(device)
            b_k_test = b_k_layers[0]
            H_test = calibration_data[first_layer].to(device).float()[:100]  # Sample

            # Original K (including bias)
            K_orig = H_test @ W_k_test.float().T
            if b_k_test is not None:
                K_orig = K_orig + b_k_test.float().to(K_orig.device)

            # Reconstructed K (decompress_k now includes bias)
            with torch.no_grad():
                c_test = test_comp.compress(H_test)
                K_recon = test_comp.decompress_k(c_test)

            # Compute relative error
            recon_error = (K_orig - K_recon).norm() / K_orig.norm()
            print(f"  Reconstruction error (layer {first_layer}): {recon_error:.4f}")
            if recon_error > 0.5:
                print(f"  WARNING: High reconstruction error! Check initialization.")

        groups[group_idx] = group

    return groups


def _compute_shared_basis_from_activations(
    K_list: List[torch.Tensor],
    V_list: List[torch.Tensor],
    d_latent: int,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Compute shared decompression basis from K/V activations.

    Stacks K/V activations across layers and computes SVD to find
    shared basis vectors that capture cross-layer structure.

    Args:
        K_list: List of K activations, each (n_tokens, d_kv)
        V_list: List of V activations, each (n_tokens, d_kv)
        d_latent: Target latent dimension

    Returns:
        Tuple of (W_uk, W_uv, k_energy, v_energy) where:
            - W_uk: Shared K decompression matrix (d_kv, d_latent)
            - W_uv: Shared V decompression matrix (d_kv, d_latent)
            - k_energy: Fraction of K variance captured
            - v_energy: Fraction of V variance captured
    """
    # Stack across layers: (group_size * n_tokens, d_kv)
    K_all = torch.cat(K_list, dim=0)
    V_all = torch.cat(V_list, dim=0)

    # SVD on transposed stacked K/V to find shared basis
    # K_all.T: (d_kv, group_size * n_tokens)
    # We want U: (d_kv, d_latent) - columns are basis vectors
    U_k, S_k, _ = torch.linalg.svd(K_all.T, full_matrices=False)
    U_v, S_v, _ = torch.linalg.svd(V_all.T, full_matrices=False)

    # Truncate to d_latent
    shared_W_uk = U_k[:, :d_latent]  # (d_kv, d_latent)
    shared_W_uv = U_v[:, :d_latent]

    # Compute energy (variance) retained
    k_energy = (S_k[:d_latent] ** 2).sum() / (S_k**2).sum()
    v_energy = (S_v[:d_latent] ** 2).sum() / (S_v**2).sum()

    return shared_W_uk, shared_W_uv, k_energy.item(), v_energy.item()


def _compute_shared_basis_from_weights(
    W_k_list: List[torch.Tensor],
    W_v_list: List[torch.Tensor],
    d_latent: int,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Compute shared decompression basis from weight matrices.

    This is a fallback when calibration data is not available.
    Stacks weight matrices across layers and computes SVD.

    Args:
        W_k_list: List of K projection weights, each (d_kv, d_model)
        W_v_list: List of V projection weights, each (d_kv, d_model)
        d_latent: Target latent dimension

    Returns:
        Tuple of (W_uk, W_uv, k_energy, v_energy)
    """
    # Stack weights: (group_size * d_kv, d_model)
    W_k_all = torch.cat([W.float() for W in W_k_list], dim=0)
    W_v_all = torch.cat([W.float() for W in W_v_list], dim=0)

    # SVD to find shared structure
    U_k, S_k, _ = torch.linalg.svd(W_k_all, full_matrices=False)
    U_v, S_v, _ = torch.linalg.svd(W_v_all, full_matrices=False)

    # Reshape U to get per-KV-dimension basis
    # U_k: (group_size * d_kv, min(group_size * d_kv, d_model))
    # We need to extract a (d_kv, d_latent) matrix

    d_kv = W_k_list[0].shape[0]
    group_size = len(W_k_list)

    # Average the basis vectors across layers
    # Reshape U_k to (group_size, d_kv, rank) and average
    rank = min(U_k.shape[1], d_latent * group_size)
    U_k_reshaped = U_k[:, :rank].reshape(group_size, d_kv, rank)
    U_v_reshaped = U_v[:, :rank].reshape(group_size, d_kv, rank)

    # Average across layers and take top d_latent components
    U_k_avg = U_k_reshaped.mean(dim=0)  # (d_kv, rank)
    U_v_avg = U_v_reshaped.mean(dim=0)

    # SVD on averaged basis to get orthonormal columns
    U_k_final, S_k_final, _ = torch.linalg.svd(U_k_avg, full_matrices=False)
    U_v_final, S_v_final, _ = torch.linalg.svd(U_v_avg, full_matrices=False)

    shared_W_uk = U_k_final[:, :d_latent]
    shared_W_uv = U_v_final[:, :d_latent]

    # Approximate energy retained
    k_energy = (S_k[:d_latent] ** 2).sum() / (S_k**2).sum()
    v_energy = (S_v[:d_latent] ** 2).sum() / (S_v**2).sum()

    return shared_W_uk, shared_W_uv, k_energy.item(), v_energy.item()


def calibrate_xkv(
    model: nn.Module,
    tokenizer,
    handler,
    config: MLAConfig,
    texts: Optional[List[str]] = None,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    num_samples: int = 128,
    max_length: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[int, XKVCompressionGroup]:
    """Calibrate xKV compression using sample text data.

    This is a convenience function that:
    1. Collects calibration data from the model
    2. Initializes xKV compression groups using cross-layer SVD

    Args:
        model: The HuggingFace model to compress
        tokenizer: Tokenizer for the model
        handler: ModelHandler for extracting weights
        config: MLAConfig with cross-layer settings
        texts: Optional list of texts for calibration
        dataset_name: HuggingFace dataset for calibration if texts not provided
        dataset_config: Dataset configuration
        num_samples: Number of calibration samples
        max_length: Maximum sequence length
        device: Device for computation

    Returns:
        Dictionary mapping group_idx to XKVCompressionGroup
    """
    from .initialization import collect_calibration_data

    if device is None:
        device = next(model.parameters()).device

    # Collect calibration data
    calibration_data = collect_calibration_data(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        num_samples=num_samples,
        max_length=max_length,
        device=device,
    )

    # Initialize compression groups
    groups = cross_layer_svd_init(
        handler=handler,
        config=config,
        calibration_data=calibration_data,
        verbose=True,
    )

    return groups
