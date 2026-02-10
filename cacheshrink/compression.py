"""MLA Compression module with Stiefel manifold constraints."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

import geoopt
from geoopt.manifolds import Stiefel

from .config import MLAConfig
from .utils import orthonormalize_columns, check_orthonormality, random_orthonormal_columns


class MLACompression(nn.Module):
    """Multi-Head Latent Attention compression module with SEPARATE K/V compression.

    Uses separate compression for K and V to preserve reconstruction quality:
    - W_down_k: Euclidean linear projection for K (d_model -> d_latent)
    - W_down_v: Euclidean linear projection for V (d_model -> d_latent)
    - W_uk: Stiefel manifold parameter for K decompression (d_kv x d_latent, orthonormal columns)
    - W_uv: Stiefel manifold parameter for V decompression (d_kv x d_latent, orthonormal columns)

    Cache stores [c_k, c_v] of total size 2*d_latent per token.
    Compression ratio = 2*d_kv / (2*d_latent) = d_kv/d_latent.

    Note: All trainable parameters are kept in float32 for numerical stability during optimization.
    """

    def __init__(self, config: MLAConfig):
        """Initialize MLA compression module.

        Args:
            config: MLA configuration
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_latent = config.computed_d_latent
        self.d_kv = config.d_kv

        # Separate compression for K and V
        self.W_down_k = nn.Linear(self.d_model, self.d_latent, bias=False)
        self.W_down_v = nn.Linear(self.d_model, self.d_latent, bias=False)

        # Stiefel manifold for decompression matrices
        self.stiefel = Stiefel(canonical=False)

        # Initialize with random orthonormal matrices
        W_uk_init = random_orthonormal_columns(self.d_kv, self.d_latent).float()
        W_uv_init = random_orthonormal_columns(self.d_kv, self.d_latent).float()

        self.W_uk = geoopt.ManifoldParameter(W_uk_init, manifold=self.stiefel)
        self.W_uv = geoopt.ManifoldParameter(W_uv_init, manifold=self.stiefel)

        # Buffers for reconstruction loss (optional, set via store_original_weights)
        self.register_buffer('W_k_original', None)
        self.register_buffer('W_v_original', None)

        # Buffers for K/V biases (required for models like Qwen that have biases)
        self.register_buffer('b_k', None)
        self.register_buffer('b_v', None)

    def store_original_weights(self, W_k: torch.Tensor, W_v: torch.Tensor) -> None:
        """Store original K/V projection weights as frozen buffers for reconstruction loss.

        Args:
            W_k: Original key projection weights of shape (d_kv, d_model)
            W_v: Original value projection weights of shape (d_kv, d_model)
        """
        self.register_buffer('W_k_original', W_k.clone().detach())
        self.register_buffer('W_v_original', W_v.clone().detach())

    def compute_reconstruction_loss(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute MSE between original and reconstructed K/V.

        Args:
            h: Hidden states of shape (batch, seq_len, d_model)

        Returns:
            Tuple of (k_loss, v_loss) - MSE losses for K and V reconstruction

        Raises:
            ValueError: If original weights have not been stored
        """
        if self.W_k_original is None or self.W_v_original is None:
            raise ValueError("Original weights not stored. Call store_original_weights() first.")

        # Compute targets using original weights (including biases if present)
        h_f32 = h.float()
        K_target = torch.matmul(h_f32, self.W_k_original.float().T)
        V_target = torch.matmul(h_f32, self.W_v_original.float().T)
        if self.b_k is not None:
            K_target = K_target + self.b_k.float()
        if self.b_v is not None:
            V_target = V_target + self.b_v.float()

        # Compute reconstructed K/V
        c_kv = self.compress(h)
        K_recon = self.decompress_k(c_kv).float()
        V_recon = self.decompress_v(c_kv).float()

        return F.mse_loss(K_recon, K_target), F.mse_loss(V_recon, V_target)

    def has_original_weights(self) -> bool:
        """Check if original weights are stored."""
        return self.W_k_original is not None and self.W_v_original is not None

    def compress(self, h: torch.Tensor) -> torch.Tensor:
        """Compress hidden states to combined latent representation [c_k, c_v].

        Args:
            h: Hidden states of shape (batch, seq_len, d_model)

        Returns:
            Latent representation of shape (batch, seq_len, 2*d_latent)
            First d_latent dims are c_k, last d_latent dims are c_v
        """
        # Use float32 for numerical stability during matmul
        h_f32 = h.float()
        c_k = torch.nn.functional.linear(h_f32, self.W_down_k.weight.float(), None)
        c_v = torch.nn.functional.linear(h_f32, self.W_down_v.weight.float(), None)
        result = torch.cat([c_k, c_v], dim=-1)
        return result.to(h.dtype)

    def decompress_k(self, c_kv: torch.Tensor) -> torch.Tensor:
        """Decompress latent representation to keys.

        Args:
            c_kv: Latent representation of shape (batch, seq_len, 2*d_latent)

        Returns:
            Keys of shape (batch, seq_len, d_kv)
        """
        c_k = c_kv[..., :self.d_latent].float()
        result = torch.matmul(c_k, self.W_uk.float().T)
        # Add K bias if present (critical for models like Qwen)
        if self.b_k is not None:
            result = result + self.b_k.to(result.device).float()
        return result.to(c_kv.dtype)

    def decompress_v(self, c_kv: torch.Tensor) -> torch.Tensor:
        """Decompress latent representation to values.

        Args:
            c_kv: Latent representation of shape (batch, seq_len, 2*d_latent)

        Returns:
            Values of shape (batch, seq_len, d_kv)
        """
        c_v = c_kv[..., self.d_latent:].float()
        result = torch.matmul(c_v, self.W_uv.float().T)
        # Add V bias if present (critical for models like Qwen)
        if self.b_v is not None:
            result = result + self.b_v.to(result.device).float()
        return result.to(c_kv.dtype)

    def store_biases(self, b_k: Optional[torch.Tensor], b_v: Optional[torch.Tensor]) -> None:
        """Store K/V biases as frozen buffers.

        Args:
            b_k: Key projection bias of shape (d_kv,) or None
            b_v: Value projection bias of shape (d_kv,) or None
        """
        if b_k is not None:
            self.register_buffer("b_k", b_k.clone().detach())
        if b_v is not None:
            self.register_buffer("b_v", b_v.clone().detach())

    def decompress(self, c_kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress latent representation to keys and values.

        Args:
            c_kv: Latent representation of shape (batch, seq_len, 2*d_latent)

        Returns:
            Tuple of (keys, values), each of shape (batch, seq_len, d_kv)
        """
        return self.decompress_k(c_kv), self.decompress_v(c_kv)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress hidden states and decompress to K, V.

        Args:
            h: Hidden states of shape (batch, seq_len, d_model)

        Returns:
            Tuple of (c_kv, K, V) where:
                - c_kv: (batch, seq_len, 2*d_latent)
                - K: (batch, seq_len, d_kv)
                - V: (batch, seq_len, d_kv)
        """
        c_kv = self.compress(h)
        K = self.decompress_k(c_kv)
        V = self.decompress_v(c_kv)
        return c_kv, K, V

    def check_orthonormality(self) -> Dict[str, Tuple[float, float]]:
        """Check orthonormality of decompression matrices.

        W_uk and W_uv have orthonormal columns (W.T @ W = I).

        Returns:
            Dictionary with 'W_uk' and 'W_uv' keys, each containing
            (max_error, mean_error) from identity matrix
        """
        return {
            "W_uk": check_orthonormality(self.W_uk, mode="columns"),
            "W_uv": check_orthonormality(self.W_uv, mode="columns"),
        }

    def project_to_manifold(self) -> None:
        """Project W_uk and W_uv back to Stiefel manifold."""
        with torch.no_grad():
            self.W_uk.data = orthonormalize_columns(self.W_uk.data)
            self.W_uv.data = orthonormalize_columns(self.W_uv.data)

    def init_from_weights(
        self,
        W_down_k: torch.Tensor,
        W_down_v: torch.Tensor,
        W_uk: torch.Tensor,
        W_uv: torch.Tensor
    ) -> None:
        """Initialize from pre-computed weights.

        Args:
            W_down_k: K compression weights of shape (d_latent, d_model)
            W_down_v: V compression weights of shape (d_latent, d_model)
            W_uk: K decompression weights of shape (d_kv, d_latent) - orthonormal columns
            W_uv: V decompression weights of shape (d_kv, d_latent) - orthonormal columns
        """
        target_dtype = self.W_down_k.weight.dtype
        self.W_down_k.weight.data = W_down_k.to(device=self.W_down_k.weight.device, dtype=target_dtype)
        self.W_down_v.weight.data = W_down_v.to(device=self.W_down_v.weight.device, dtype=target_dtype)
        self.W_uk.data = W_uk.to(device=self.W_uk.device, dtype=self.W_uk.dtype)
        self.W_uv.data = W_uv.to(device=self.W_uv.device, dtype=self.W_uv.dtype)

    def get_euclidean_params(self):
        """Get Euclidean (non-manifold) parameters for standard optimizer."""
        return [self.W_down_k.weight, self.W_down_v.weight]

    def get_manifold_params(self):
        """Get manifold parameters for Riemannian optimizer."""
        return [self.W_uk, self.W_uv]

    def _apply(self, fn):
        """Override _apply to keep trainable parameters in float32."""
        super()._apply(fn)
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
            f"d_model={self.d_model}, d_latent={self.d_latent}, d_kv={self.d_kv}, "
            f"compression_ratio={self.d_kv / self.d_latent:.1f}x, "
            f"cache_size=2*d_latent={2*self.d_latent}"
        )
