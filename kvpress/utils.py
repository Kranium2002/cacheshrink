"""Utility functions for MLA operations."""

import torch
from typing import Tuple, Optional


def orthonormalize_rows(W: torch.Tensor, method: str = "polar") -> torch.Tensor:
    """Orthonormalize the rows of a matrix.

    Given a matrix W of shape (m, n) where m <= n, this produces a matrix
    with orthonormal rows (W @ W.T = I).

    Args:
        W: Matrix of shape (m, n) where m <= n
        method: "qr" for QR decomposition, "polar" for polar decomposition.
                "polar" finds the CLOSEST orthonormal matrix (Procrustes solution).

    Returns:
        Matrix with orthonormal rows of same shape as W
    """
    m, n = W.shape
    if m > n:
        raise ValueError(f"Cannot orthonormalize rows when m > n: got shape {W.shape}")

    if method == "qr":
        # QR decomposition of W.T gives us Q with orthonormal columns
        # So Q.T has orthonormal rows
        Q, _ = torch.linalg.qr(W.T, mode="reduced")
        return Q.T
    else:
        # Polar decomposition: find closest orthonormal matrix
        # For W = U @ S @ Vh, the closest orthonormal matrix is U @ Vh
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return U @ Vh


def orthonormalize_columns(W: torch.Tensor, method: str = "polar") -> torch.Tensor:
    """Orthonormalize the columns of a matrix.

    Given a matrix W of shape (m, n) where n <= m, this produces a matrix
    with orthonormal columns (W.T @ W = I).

    Args:
        W: Matrix of shape (m, n) where n <= m
        method: "qr" for QR decomposition, "polar" for polar decomposition.
                "polar" finds the CLOSEST orthonormal matrix (Procrustes solution)
                which is better for preserving the original matrix's direction.

    Returns:
        Matrix with orthonormal columns of same shape as W
    """
    m, n = W.shape
    if n > m:
        raise ValueError(f"Cannot orthonormalize columns when n > m: got shape {W.shape}")

    if method == "qr":
        Q, _ = torch.linalg.qr(W, mode="reduced")
        return Q
    else:
        # Polar decomposition: find closest orthonormal matrix
        # For W = U @ S @ Vh, the closest orthonormal matrix is U @ Vh
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return U @ Vh


def check_orthonormality(W: torch.Tensor, mode: str = "rows") -> Tuple[float, float]:
    """Check how close a matrix is to having orthonormal rows/columns.

    Args:
        W: Matrix to check
        mode: "rows" to check W @ W.T = I, "columns" to check W.T @ W = I

    Returns:
        Tuple of (max_error, mean_error) from identity matrix
    """
    if mode == "rows":
        gram = W @ W.T
    else:
        gram = W.T @ W

    identity = torch.eye(gram.shape[0], device=W.device, dtype=W.dtype)
    diff = gram - identity
    max_error = diff.abs().max().item()
    mean_error = diff.abs().mean().item()
    return max_error, mean_error


def random_orthonormal_rows(m: int, n: int, device: torch.device = None,
                            dtype: torch.dtype = None) -> torch.Tensor:
    """Generate a random matrix with orthonormal rows.

    Args:
        m: Number of rows
        n: Number of columns (must be >= m)
        device: Torch device
        dtype: Torch dtype

    Returns:
        Matrix of shape (m, n) with orthonormal rows
    """
    if m > n:
        raise ValueError(f"Cannot create orthonormal rows when m > n: m={m}, n={n}")

    # Generate random matrix and orthonormalize
    W = torch.randn(m, n, device=device, dtype=dtype)
    return orthonormalize_rows(W)


def random_orthonormal_columns(m: int, n: int, device: torch.device = None,
                               dtype: torch.dtype = None) -> torch.Tensor:
    """Generate a random matrix with orthonormal columns.

    Args:
        m: Number of rows (must be >= n)
        n: Number of columns
        device: Torch device
        dtype: Torch dtype

    Returns:
        Matrix of shape (m, n) with orthonormal columns
    """
    if n > m:
        raise ValueError(f"Cannot create orthonormal columns when n > m: m={m}, n={n}")

    # Generate random matrix and orthonormalize
    W = torch.randn(m, n, device=device, dtype=dtype)
    return orthonormalize_columns(W)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for Grouped Query Attention.

    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_kv_heads, seq_len, head_dim) to
    (batch, num_attention_heads, seq_len, head_dim).

    Args:
        hidden_states: KV tensor of shape (batch, n_kv_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each KV head

    Returns:
        Expanded tensor of shape (batch, n_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def get_device(model_or_tensor) -> torch.device:
    """Get device from a model or tensor."""
    if hasattr(model_or_tensor, "device"):
        return model_or_tensor.device
    elif hasattr(model_or_tensor, "parameters"):
        try:
            return next(model_or_tensor.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def get_dtype(model_or_tensor) -> torch.dtype:
    """Get dtype from a model or tensor."""
    if hasattr(model_or_tensor, "dtype"):
        return model_or_tensor.dtype
    elif hasattr(model_or_tensor, "parameters"):
        try:
            return next(model_or_tensor.parameters()).dtype
        except StopIteration:
            return torch.float32
    else:
        return torch.float32


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count parameters in a module.

    Args:
        module: PyTorch module
        trainable_only: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def format_size(num_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"
