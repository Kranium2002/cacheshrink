"""Fast SVD implementations for MLA initialization."""

import torch
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import threading


def randomized_svd(
    A: torch.Tensor,
    k: int,
    n_oversamples: int = 10,
    n_power_iterations: int = 2,
    fallback_to_full: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized SVD for fast truncated SVD computation.

    Uses the algorithm from Halko et al. "Finding Structure with Randomness"

    Args:
        A: Input matrix of shape (m, n)
        k: Target rank (number of singular values to compute)
        n_oversamples: Extra dimensions for better accuracy (default 10)
        n_power_iterations: Power iterations for better accuracy (default 2)
        fallback_to_full: If True, fall back to full SVD on numerical errors

    Returns:
        U: Left singular vectors (m, k)
        S: Singular values (k,)
        Vh: Right singular vectors (k, n)
    """
    m, n = A.shape
    device = A.device
    dtype = A.dtype

    try:
        # Target rank with oversampling
        l = min(k + n_oversamples, min(m, n))

        # Step 1: Random projection
        # Generate random matrix and project A onto lower dimension
        Omega = torch.randn(n, l, device=device, dtype=dtype)
        Y = A @ Omega  # (m, l)

        # Step 2: Power iterations for better accuracy
        # This helps when singular values decay slowly
        for _ in range(n_power_iterations):
            # Reorthogonalize to prevent numerical issues
            Y, _ = torch.linalg.qr(Y, mode='reduced')
            Y = A @ (A.T @ Y)

        # Step 3: Orthonormalize the basis
        Q, _ = torch.linalg.qr(Y, mode='reduced')  # (m, l)

        # Step 4: Project A onto the basis and compute SVD of smaller matrix
        B = Q.T @ A  # (l, n)

        # SVD of the smaller matrix
        U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)

        # Step 5: Recover U
        U = Q @ U_hat  # (m, l)

        # Truncate to k
        return U[:, :k], S[:k], Vh[:k, :]

    except (RuntimeError, torch._C._LinAlgError) as e:
        if fallback_to_full:
            # Fall back to full SVD
            import warnings
            warnings.warn(f"Randomized SVD failed ({e}), falling back to full SVD")
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            return U[:, :k], S[:k], Vh[:k, :]
        else:
            raise


def parallel_svd_layers(
    weight_pairs: list,  # List of (W_k, W_v, calibration_data) tuples
    d_latent: int,
    use_randomized: bool = True,
    max_workers: int = 4,
    verbose: bool = True,
) -> list:
    """Compute SVD for multiple layers in parallel.

    Uses thread pool to overlap CPU/GPU work and potentially run
    multiple SVDs concurrently.

    Args:
        weight_pairs: List of (W_k, W_v, calibration_data_or_None) per layer
        d_latent: Target latent dimension
        use_randomized: Whether to use randomized SVD
        max_workers: Number of parallel workers
        verbose: Print progress

    Returns:
        List of (W_down, W_uk, W_uv) tuples per layer
    """
    from .utils import orthonormalize_rows

    results = [None] * len(weight_pairs)
    lock = threading.Lock()
    completed = [0]

    def process_layer(idx, W_k, W_v, calibration_data):
        """Process a single layer."""
        device = W_k.device
        dtype = W_k.dtype
        d_kv, d_model = W_k.shape

        # Use float32 for computation
        compute_dtype = torch.float32

        # Compute scaling from calibration or weights
        if calibration_data is not None:
            calibration_f32 = calibration_data.to(compute_dtype)
            W_k_f32 = W_k.to(compute_dtype)
            W_v_f32 = W_v.to(compute_dtype)

            with torch.no_grad():
                K_out = calibration_f32 @ W_k_f32.T
                V_out = calibration_f32 @ W_v_f32.T
                k_norm = K_out.pow(2).mean().sqrt()
                v_norm = V_out.pow(2).mean().sqrt()

            if k_norm > 0 and v_norm > 0:
                scale_k = torch.sqrt(v_norm / k_norm)
                scale_v = torch.sqrt(k_norm / v_norm)
            else:
                scale_k = torch.tensor(1.0, device=device, dtype=compute_dtype)
                scale_v = torch.tensor(1.0, device=device, dtype=compute_dtype)
        else:
            W_k_f32 = W_k.to(compute_dtype)
            W_v_f32 = W_v.to(compute_dtype)
            k_norm = W_k_f32.norm()
            v_norm = W_v_f32.norm()
            if k_norm > 0 and v_norm > 0:
                scale_k = torch.sqrt(v_norm / k_norm)
                scale_v = torch.sqrt(k_norm / v_norm)
            else:
                scale_k = torch.tensor(1.0, device=device, dtype=compute_dtype)
                scale_v = torch.tensor(1.0, device=device, dtype=compute_dtype)

        # Scale and stack
        W_k_scaled = W_k_f32 * scale_k
        W_v_scaled = W_v_f32 * scale_v
        W_joint = torch.cat([W_k_scaled, W_v_scaled], dim=0)

        # SVD
        if use_randomized:
            U, S, Vh = randomized_svd(W_joint, d_latent, n_oversamples=20, n_power_iterations=2)
            U_trunc, S_trunc, Vh_trunc = U, S, Vh
        else:
            U, S, Vh = torch.linalg.svd(W_joint, full_matrices=False)
            U_trunc = U[:, :d_latent]
            S_trunc = S[:d_latent]
            Vh_trunc = Vh[:d_latent, :]

        # Compute W_down, W_up
        sqrt_S = torch.sqrt(S_trunc)
        W_down = sqrt_S.unsqueeze(1) * Vh_trunc
        W_up = U_trunc * sqrt_S.unsqueeze(0)

        # Split and undo scaling
        W_uk_raw = W_up[:d_kv, :].T / scale_k
        W_uv_raw = W_up[d_kv:, :].T / scale_v

        # Orthonormalize
        W_uk = orthonormalize_rows(W_uk_raw)
        W_uv = orthonormalize_rows(W_uv_raw)

        # Convert back to original dtype
        result = (W_down.to(dtype), W_uk.to(dtype), W_uv.to(dtype))

        with lock:
            results[idx] = result
            completed[0] += 1
            if verbose:
                print(f"\rProcessed layer {completed[0]}/{len(weight_pairs)}", end="", flush=True)

        return result

    # Process layers in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, (W_k, W_v, calib) in enumerate(weight_pairs):
            future = executor.submit(process_layer, idx, W_k, W_v, calib)
            futures.append(future)

        # Wait for all to complete
        for future in futures:
            future.result()

    if verbose:
        print()  # Newline after progress

    return results


def batched_randomized_svd(
    matrices: list,  # List of tensors
    k: int,
    n_oversamples: int = 10,
    n_power_iterations: int = 2,
) -> list:
    """Process multiple matrices with randomized SVD using CUDA streams.

    Creates separate CUDA streams for each matrix to allow some overlap.

    Args:
        matrices: List of input matrices (can be different sizes)
        k: Target rank
        n_oversamples: Extra dimensions for accuracy
        n_power_iterations: Power iterations

    Returns:
        List of (U, S, Vh) tuples
    """
    if not matrices:
        return []

    device = matrices[0].device
    n_matrices = len(matrices)

    # Create CUDA streams
    streams = [torch.cuda.Stream(device=device) for _ in range(n_matrices)]
    results = [None] * n_matrices

    # Launch SVD on each stream
    for i, (matrix, stream) in enumerate(zip(matrices, streams)):
        with torch.cuda.stream(stream):
            results[i] = randomized_svd(matrix, k, n_oversamples, n_power_iterations)

    # Synchronize all streams
    for stream in streams:
        stream.synchronize()

    return results
