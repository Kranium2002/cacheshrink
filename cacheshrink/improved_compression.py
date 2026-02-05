"""Improved MLA compression techniques.

Key improvements over basic SVD:
1. Calibration-aware SVD - finds optimal subspace from actual K/V outputs
2. Decoupled RoPE - separates positional encoding from compressed content
3. Joint K-V compression - single latent for both K and V (more efficient)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List, TYPE_CHECKING
from tqdm import tqdm

from .utils import orthonormalize_columns

if TYPE_CHECKING:
    from .config import MLAConfig


def calibration_aware_svd(
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    hidden_states: torch.Tensor,
    d_latent: int,
    max_samples: int = 10000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Calibration-aware SVD for better compression.

    Instead of SVD on W_k/W_v directly, we do SVD on the actual K/V outputs:
    - K = H @ W_k.T  (what the model actually computes)
    - SVD(K) finds the principal directions in K-space that matter for this data

    This typically captures 10-20% more energy than weight-only SVD because
    the hidden state distribution concentrates K/V in a lower-dimensional subspace.

    Args:
        W_k: Key projection weights (d_kv, d_model)
        W_v: Value projection weights (d_kv, d_model)
        hidden_states: Calibration hidden states (n_tokens, d_model)
        d_latent: Target latent dimension
        max_samples: Maximum calibration samples to use

    Returns:
        Tuple of (W_down_k, W_down_v, W_uk, W_uv, stats)
    """
    device = W_k.device
    dtype = W_k.dtype

    # Subsample if needed
    if hidden_states.size(0) > max_samples:
        indices = torch.randperm(hidden_states.size(0))[:max_samples]
        hidden_states = hidden_states[indices]

    # Move to same device and use float32 for SVD
    H = hidden_states.to(device).float()
    W_k_f32 = W_k.float()
    W_v_f32 = W_v.float()

    # Compute actual K and V outputs
    K = H @ W_k_f32.T  # (n_tokens, d_kv)
    V = H @ W_v_f32.T  # (n_tokens, d_kv)

    # SVD on K (transposed so we get right singular vectors as columns)
    # K.T has shape (d_kv, n_tokens)
    # We want the principal directions in d_kv space
    U_k, S_k, _ = torch.linalg.svd(K.T, full_matrices=False)
    U_v, S_v, _ = torch.linalg.svd(V.T, full_matrices=False)

    # Take top d_latent components
    W_uk = U_k[:, :d_latent]  # (d_kv, d_latent)
    W_uv = U_v[:, :d_latent]  # (d_kv, d_latent)

    # Compute energy captured
    energy_k = (S_k[:d_latent]**2).sum() / (S_k**2).sum()
    energy_v = (S_v[:d_latent]**2).sum() / (S_v**2).sum()

    # Compute optimal W_down: minimize ||K - (H @ W_down.T) @ W_u.T||
    # Solution: W_down = W_u.T @ W_k (same as before, but W_u is calibration-optimized)
    W_down_k = W_uk.T @ W_k_f32  # (d_latent, d_model)
    W_down_v = W_uv.T @ W_v_f32  # (d_latent, d_model)

    # Compute reconstruction error
    K_recon = (H @ W_down_k.T) @ W_uk.T
    V_recon = (H @ W_down_v.T) @ W_uv.T
    recon_err_k = torch.norm(K - K_recon) / torch.norm(K)
    recon_err_v = torch.norm(V - V_recon) / torch.norm(V)

    stats = {
        "k_energy": energy_k.item(),
        "v_energy": energy_v.item(),
        "k_recon_error": recon_err_k.item(),
        "v_recon_error": recon_err_v.item(),
    }

    return (
        W_down_k.to(dtype),
        W_down_v.to(dtype),
        W_uk.to(dtype),
        W_uv.to(dtype),
        stats
    )


def joint_kv_svd(
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    hidden_states: torch.Tensor,
    d_latent: int,
    max_samples: int = 10000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Joint K-V compression with shared latent (like DeepSeek MLA).

    Uses a SINGLE latent c that reconstructs both K and V:
    - c = H @ W_down.T
    - K = c @ W_uk.T
    - V = c @ W_uv.T

    This is more efficient (single latent instead of [c_k, c_v]) and
    allows the latent to capture shared structure between K and V.

    Args:
        W_k: Key projection weights (d_kv, d_model)
        W_v: Value projection weights (d_kv, d_model)
        hidden_states: Calibration hidden states (n_tokens, d_model)
        d_latent: Target latent dimension
        max_samples: Maximum calibration samples

    Returns:
        Tuple of (W_down, W_uk, W_uv, stats)
        - W_down: (d_latent, d_model) shared compression
        - W_uk: (d_kv, d_latent) K decompression
        - W_uv: (d_kv, d_latent) V decompression
    """
    device = W_k.device
    dtype = W_k.dtype

    if hidden_states.size(0) > max_samples:
        indices = torch.randperm(hidden_states.size(0))[:max_samples]
        hidden_states = hidden_states[indices]

    H = hidden_states.to(device).float()
    W_k_f32 = W_k.float()
    W_v_f32 = W_v.float()

    # Compute K and V
    K = H @ W_k_f32.T  # (n_tokens, d_kv)
    V = H @ W_v_f32.T  # (n_tokens, d_kv)

    # Stack K and V for joint SVD
    # This finds directions that are important for BOTH K and V
    KV = torch.cat([K, V], dim=1)  # (n_tokens, 2*d_kv)

    # SVD on hidden states weighted by importance to K and V
    # Alternative: SVD on H directly, weighted by how much each direction affects KV
    # H.T @ H gives covariance of hidden states
    # We want directions in H that most affect KV

    # Compute H.T @ KV to find which H directions matter for KV
    H_to_KV = H.T @ KV  # (d_model, 2*d_kv)

    # SVD to find optimal compression directions in H space
    U_h, S_h, Vh = torch.linalg.svd(H_to_KV, full_matrices=False)

    # W_down projects H to latent space using top directions
    W_down = U_h[:, :d_latent].T  # (d_latent, d_model)

    # Compute latent representations
    C = H @ W_down.T  # (n_tokens, d_latent)

    # Find optimal W_uk and W_uv to reconstruct K and V from C
    # Least squares: K ≈ C @ W_uk.T => W_uk.T = (C.T @ C)^{-1} @ C.T @ K
    CtC_inv = torch.linalg.inv(C.T @ C + 1e-6 * torch.eye(d_latent, device=device))
    W_uk = (CtC_inv @ C.T @ K).T  # (d_kv, d_latent)
    W_uv = (CtC_inv @ C.T @ V).T  # (d_kv, d_latent)

    # Orthonormalize W_uk and W_uv for Stiefel manifold
    W_uk = orthonormalize_columns(W_uk)
    W_uv = orthonormalize_columns(W_uv)

    # Keep the original W_down from SVD - it was optimized for joint K/V reconstruction
    # and changing it after orthonormalization often makes things worse.
    # The orthonormalization of W_uk/W_uv will cause some reconstruction loss,
    # but that's the price we pay for Stiefel manifold constraint.

    # Reconstruction (using original C from before orthonormalization - just for stats)
    K_recon = C @ W_uk.T
    V_recon = C @ W_uv.T

    recon_err_k = torch.norm(K - K_recon) / torch.norm(K)
    recon_err_v = torch.norm(V - V_recon) / torch.norm(V)

    stats = {
        "k_recon_error": recon_err_k.item(),
        "v_recon_error": recon_err_v.item(),
        "latent_dim": d_latent,
        "compression_ratio": (2 * W_k.shape[0]) / d_latent,  # 2*d_kv / d_latent
    }

    return W_down.to(dtype), W_uk.to(dtype), W_uv.to(dtype), stats


class JointKVCompression(nn.Module):
    """Joint K-V compression with shared latent (DeepSeek MLA style).

    Uses a SINGLE shared latent c that reconstructs both K and V:
    - c = H @ W_down.T
    - K = c @ W_uk.T
    - V = c @ W_uv.T

    This is more efficient than separate compression:
    - Cache stores only c (d_latent) instead of [c_k, c_v] (2*d_latent)
    - For the same cache size, can use 2x larger d_latent

    Trade-off: K and V must share the same compressed representation,
    which may slightly reduce reconstruction quality.
    """

    def __init__(self, config: "MLAConfig"):
        """Initialize joint K-V compression.

        Args:
            config: MLA configuration
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_latent = config.computed_d_latent
        self.d_kv = config.d_kv

        # Shared compression (single projection for both K and V)
        self.W_down = nn.Linear(self.d_model, self.d_latent, bias=False)

        # Separate decompression on Stiefel manifold
        import geoopt
        from geoopt.manifolds import Stiefel

        self.stiefel = Stiefel(canonical=False)

        W_uk_init = orthonormalize_columns(torch.randn(self.d_kv, self.d_latent))
        W_uv_init = orthonormalize_columns(torch.randn(self.d_kv, self.d_latent))

        self.W_uk = geoopt.ManifoldParameter(W_uk_init.float(), manifold=self.stiefel)
        self.W_uv = geoopt.ManifoldParameter(W_uv_init.float(), manifold=self.stiefel)

    def compress(self, h: torch.Tensor) -> torch.Tensor:
        """Compress hidden states to shared latent.

        Args:
            h: Hidden states of shape (batch, seq_len, d_model)

        Returns:
            Latent representation of shape (batch, seq_len, d_latent)
        """
        # Cast input to match weight dtype (weights kept float32 for Riemannian optimization)
        original_dtype = h.dtype
        h_float = h.to(self.W_down.weight.dtype)
        return self.W_down(h_float).to(original_dtype)

    def decompress_k(self, c: torch.Tensor) -> torch.Tensor:
        """Decompress latent to keys."""
        # W_uk is kept float32 for Riemannian optimization
        return torch.matmul(c.float(), self.W_uk.float().T).to(c.dtype)

    def decompress_v(self, c: torch.Tensor) -> torch.Tensor:
        """Decompress latent to values."""
        # W_uv is kept float32 for Riemannian optimization
        return torch.matmul(c.float(), self.W_uv.float().T).to(c.dtype)

    def decompress(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress latent to keys and values."""
        return self.decompress_k(c), self.decompress_v(c)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress and decompress.

        Returns:
            Tuple of (c, K, V)
        """
        c = self.compress(h)
        K = self.decompress_k(c)
        V = self.decompress_v(c)
        return c, K, V

    def check_orthonormality(self) -> Dict[str, Tuple[float, float]]:
        """Check orthonormality of decompression matrices."""
        from .utils import check_orthonormality
        return {
            "W_uk": check_orthonormality(self.W_uk, mode="columns"),
            "W_uv": check_orthonormality(self.W_uv, mode="columns"),
        }

    def init_from_weights(
        self,
        W_down: torch.Tensor,
        W_uk: torch.Tensor,
        W_uv: torch.Tensor,
    ) -> None:
        """Initialize from pre-computed weights."""
        self.W_down.weight.data = W_down.to(self.W_down.weight.device).float()
        self.W_uk.data = W_uk.to(self.W_uk.device).float()
        self.W_uv.data = W_uv.to(self.W_uv.device).float()

    @property
    def cache_dim(self) -> int:
        """Dimension of compressed cache per token."""
        return self.d_latent  # Just c, not [c_k, c_v]

    @property
    def compression_ratio(self) -> float:
        """Effective compression ratio."""
        return (2 * self.d_kv) / self.d_latent

    def _apply(self, fn):
        """Override _apply to keep trainable parameters in float32.

        This prevents manifold parameters from being converted to float16
        when model.to(dtype) is called, which would break Riemannian optimization
        since geoopt's QR decomposition doesn't support half precision.
        """
        super()._apply(fn)
        if self.W_down.weight.dtype != torch.float32:
            self.W_down.weight.data = self.W_down.weight.data.float()
        if self.W_uk.dtype != torch.float32:
            self.W_uk.data = self.W_uk.data.float()
        if self.W_uv.dtype != torch.float32:
            self.W_uv.data = self.W_uv.data.float()
        return self

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_latent={self.d_latent}, d_kv={self.d_kv}, "
            f"compression_ratio={self.compression_ratio:.1f}x (joint), "
            f"cache_size=d_latent={self.d_latent}"
        )


class DecoupledRoPECompression(nn.Module):
    """MLA Compression with decoupled RoPE (DeepSeek-style).

    Key insight: RoPE encodes position, which is hard to compress.
    Solution: Split K into:
    - K_rope: Small component that carries RoPE (not compressed)
    - K_content: Larger component for content (compressed)

    Final K = [K_rope, K_content_decompressed]

    This preserves positional information while still compressing content.
    """

    def __init__(
        self,
        d_model: int,
        d_kv: int,
        d_latent: int,
        d_rope: int = 64,  # Dimension for RoPE (not compressed)
    ):
        super().__init__()
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_latent = d_latent
        self.d_rope = d_rope
        self.d_content = d_kv - d_rope  # Content dimension

        # RoPE projection (not compressed) - small
        self.W_rope_k = nn.Linear(d_model, d_rope, bias=False)
        self.W_rope_v = nn.Linear(d_model, d_rope, bias=False)  # V also gets some uncompressed

        # Content compression
        self.W_down_k = nn.Linear(d_model, d_latent, bias=False)
        self.W_down_v = nn.Linear(d_model, d_latent, bias=False)

        # Content decompression (Stiefel manifold)
        import geoopt
        from geoopt.manifolds import Stiefel

        self.stiefel = Stiefel(canonical=False)
        W_uk_init = torch.randn(self.d_content, d_latent)
        W_uv_init = torch.randn(self.d_content, d_latent)
        W_uk_init = orthonormalize_columns(W_uk_init).float()
        W_uv_init = orthonormalize_columns(W_uv_init).float()

        self.W_uk = geoopt.ManifoldParameter(W_uk_init, manifold=self.stiefel)
        self.W_uv = geoopt.ManifoldParameter(W_uv_init, manifold=self.stiefel)

    def compress(self, h: torch.Tensor) -> torch.Tensor:
        """Compress hidden states.

        Returns: (batch, seq, d_rope*2 + d_latent*2)
        - First d_rope: K_rope (uncompressed positional)
        - Next d_rope: V_rope (uncompressed)
        - Next d_latent: c_k (compressed K content)
        - Last d_latent: c_v (compressed V content)
        """
        # Cast input to match weight dtype (weights kept float32 for Riemannian optimization)
        original_dtype = h.dtype
        h_float = h.to(self.W_rope_k.weight.dtype)
        k_rope = self.W_rope_k(h_float)
        v_rope = self.W_rope_v(h_float)
        c_k = self.W_down_k(h_float)
        c_v = self.W_down_v(h_float)

        return torch.cat([k_rope, v_rope, c_k, c_v], dim=-1).to(original_dtype)

    def decompress_k(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress to full K."""
        k_rope = compressed[..., :self.d_rope]
        c_k = compressed[..., 2*self.d_rope:2*self.d_rope+self.d_latent]

        # W_uk is kept float32 for Riemannian optimization
        k_content = torch.matmul(c_k.float(), self.W_uk.float().T).to(compressed.dtype)
        return torch.cat([k_rope, k_content], dim=-1)

    def decompress_v(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress to full V."""
        v_rope = compressed[..., self.d_rope:2*self.d_rope]
        c_v = compressed[..., 2*self.d_rope+self.d_latent:]

        # W_uv is kept float32 for Riemannian optimization
        v_content = torch.matmul(c_v.float(), self.W_uv.float().T).to(compressed.dtype)
        return torch.cat([v_rope, v_content], dim=-1)

    @property
    def cache_dim(self) -> int:
        """Dimension of compressed cache per token."""
        return 2 * self.d_rope + 2 * self.d_latent

    @property
    def compression_ratio(self) -> float:
        """Effective compression ratio."""
        original = 2 * self.d_kv
        compressed = self.cache_dim
        return original / compressed

    def check_orthonormality(self) -> Dict[str, Tuple[float, float]]:
        """Check orthonormality of decompression matrices."""
        from .utils import check_orthonormality
        return {
            "W_uk": check_orthonormality(self.W_uk, mode="columns"),
            "W_uv": check_orthonormality(self.W_uv, mode="columns"),
        }

    def decompress(self, compressed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress to keys and values."""
        return self.decompress_k(compressed), self.decompress_v(compressed)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress and decompress."""
        c = self.compress(h)
        K = self.decompress_k(c)
        V = self.decompress_v(c)
        return c, K, V

    def _apply(self, fn):
        """Override _apply to keep trainable parameters in float32.

        This prevents manifold parameters from being converted to float16
        when model.to(dtype) is called, which would break Riemannian optimization
        since geoopt's QR decomposition doesn't support half precision.
        """
        super()._apply(fn)
        if self.W_rope_k.weight.dtype != torch.float32:
            self.W_rope_k.weight.data = self.W_rope_k.weight.data.float()
        if self.W_rope_v.weight.dtype != torch.float32:
            self.W_rope_v.weight.data = self.W_rope_v.weight.data.float()
        if self.W_down_k.weight.dtype != torch.float32:
            self.W_down_k.weight.data = self.W_down_k.weight.data.float()
        if self.W_down_v.weight.dtype != torch.float32:
            self.W_down_v.weight.data = self.W_down_v.weight.data.float()
        if self.W_uk.dtype != torch.float32:
            self.W_uk.data = self.W_uk.data.float()
        if self.W_uv.dtype != torch.float32:
            self.W_uv.data = self.W_uv.data.float()
        return self

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_kv={self.d_kv}, d_rope={self.d_rope}, "
            f"d_latent={self.d_latent}, compression_ratio={self.compression_ratio:.1f}x"
        )


def init_decoupled_rope_from_weights(
    compression: DecoupledRoPECompression,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    hidden_states: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Initialize decoupled RoPE compression from original weights.

    Strategy:
    1. First d_rope dimensions of W_k go to W_rope_k (preserves RoPE)
    2. Remaining dimensions compressed via calibration-aware SVD
    """
    d_rope = compression.d_rope
    d_latent = compression.d_latent
    device = W_k.device
    dtype = W_k.dtype

    # Split weights into RoPE and content parts
    # Assuming first d_rope dimensions handle RoPE
    W_k_rope = W_k[:d_rope, :]  # (d_rope, d_model)
    W_v_rope = W_v[:d_rope, :]
    W_k_content = W_k[d_rope:, :]  # (d_content, d_model)
    W_v_content = W_v[d_rope:, :]

    # Set RoPE projections directly (keep float32 for training stability)
    compression.W_rope_k.weight.data = W_k_rope.clone().float()
    compression.W_rope_v.weight.data = W_v_rope.clone().float()

    # Initialize content compression
    if hidden_states is not None:
        # Calibration-aware SVD for content
        H = hidden_states.to(device).float()

        K_content = H @ W_k_content.float().T
        V_content = H @ W_v_content.float().T

        U_k, S_k, _ = torch.linalg.svd(K_content.T, full_matrices=False)
        U_v, S_v, _ = torch.linalg.svd(V_content.T, full_matrices=False)

        W_uk = U_k[:, :d_latent]
        W_uv = U_v[:, :d_latent]

        energy_k = (S_k[:d_latent]**2).sum() / (S_k**2).sum()
        energy_v = (S_v[:d_latent]**2).sum() / (S_v**2).sum()
    else:
        # Weight-only SVD
        U_k, S_k, _ = torch.linalg.svd(W_k_content.float(), full_matrices=False)
        U_v, S_v, _ = torch.linalg.svd(W_v_content.float(), full_matrices=False)

        W_uk = U_k[:, :d_latent]
        W_uv = U_v[:, :d_latent]

        energy_k = (S_k[:d_latent]**2).sum() / (S_k**2).sum()
        energy_v = (S_v[:d_latent]**2).sum() / (S_v**2).sum()

    # Set content compression weights
    W_down_k = W_uk.T @ W_k_content.float()
    W_down_v = W_uv.T @ W_v_content.float()

    # Keep trainable parameters in float32 for Riemannian optimization stability
    compression.W_down_k.weight.data = W_down_k.float()
    compression.W_down_v.weight.data = W_down_v.float()
    compression.W_uk.data = W_uk.float()
    compression.W_uv.data = W_uv.float()

    return {
        "k_content_energy": energy_k.item(),
        "v_content_energy": energy_v.item(),
        "rope_dim": d_rope,
        "content_latent_dim": d_latent,
        "effective_compression": compression.compression_ratio,
    }


def compare_compression_methods(
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    hidden_states: torch.Tensor,
    d_latent: int,
) -> Dict[str, Dict[str, float]]:
    """Compare different compression methods.

    Returns reconstruction errors and energy captured for:
    1. Weight-only SVD (baseline)
    2. Calibration-aware SVD
    3. Joint K-V SVD
    """
    from .initialization import balanced_svd_init

    results = {}

    # 1. Weight-only SVD (baseline)
    W_down_k, W_down_v, W_uk, W_uv = balanced_svd_init(
        W_k, W_v, d_latent, calibration_data=None, use_randomized_svd=False
    )

    # Compute reconstruction error with calibration data
    H = hidden_states.float().to(W_k.device)
    K = H @ W_k.float().T
    V = H @ W_v.float().T

    K_recon = (H @ W_down_k.float().T) @ W_uk.float().T
    V_recon = (H @ W_down_v.float().T) @ W_uv.float().T

    results["weight_only_svd"] = {
        "k_recon_error": (torch.norm(K - K_recon) / torch.norm(K)).item(),
        "v_recon_error": (torch.norm(V - V_recon) / torch.norm(V)).item(),
    }

    # 2. Calibration-aware SVD
    W_down_k, W_down_v, W_uk, W_uv, stats = calibration_aware_svd(
        W_k, W_v, hidden_states, d_latent
    )
    results["calibration_aware_svd"] = {
        "k_recon_error": stats["k_recon_error"],
        "v_recon_error": stats["v_recon_error"],
        "k_energy": stats["k_energy"],
        "v_energy": stats["v_energy"],
    }

    # 3. Joint K-V SVD
    W_down, W_uk, W_uv, stats = joint_kv_svd(
        W_k, W_v, hidden_states, d_latent
    )
    results["joint_kv_svd"] = {
        "k_recon_error": stats["k_recon_error"],
        "v_recon_error": stats["v_recon_error"],
    }

    return results
