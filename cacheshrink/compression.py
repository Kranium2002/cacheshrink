"""MLA Compression module with Stiefel manifold constraints."""

import torch
import torch.nn as nn
from typing import Dict, Tuple

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

    def compress(self, h: torch.Tensor) -> torch.Tensor:
        """Compress hidden states to combined latent representation [c_k, c_v].

        Args:
            h: Hidden states of shape (batch, seq_len, d_model)

        Returns:
            Latent representation of shape (batch, seq_len, 2*d_latent)
            First d_latent dims are c_k, last d_latent dims are c_v
        """
        # Always use float32 for numerical stability
        h_f32 = h.float()
        c_k = self.W_down_k(h_f32)
        c_v = self.W_down_v(h_f32)
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
        result = torch.matmul(c_k, self.W_uk.T)
        return result.to(c_kv.dtype)

    def decompress_v(self, c_kv: torch.Tensor) -> torch.Tensor:
        """Decompress latent representation to values.

        Args:
            c_kv: Latent representation of shape (batch, seq_len, 2*d_latent)

        Returns:
            Values of shape (batch, seq_len, d_kv)
        """
        c_v = c_kv[..., self.d_latent:].float()
        result = torch.matmul(c_v, self.W_uv.T)
        return result.to(c_kv.dtype)

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
        self.W_down_k.weight.data = W_down_k.to(self.W_down_k.weight.device).float()
        self.W_down_v.weight.data = W_down_v.to(self.W_down_v.weight.device).float()
        self.W_uk.data = W_uk.to(self.W_uk.device).float()
        self.W_uv.data = W_uv.to(self.W_uv.device).float()

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
