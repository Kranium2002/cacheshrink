"""Cross-layer KV compression (xKV) module.

xKV compresses KV cache across multiple layers by finding shared basis vectors
using cross-layer SVD. This is particularly effective for GQA models where
per-layer redundancy is already reduced.

Key insight: Adjacent layers' KV caches have high similarity in their dominant
singular vectors. By grouping layers and applying SVD across them, we find
shared basis vectors that efficiently represent all layers in the group.

Architecture:
    Standard (per-layer): Each layer has its own W_uk, W_uv matrices
    xKV (cross-layer): Group of layers SHARE W_uk, W_uv, only W_down differs

    Group [L0, L1, L2, L3]:
        Stack KVs -> Cross-layer SVD -> Shared basis U + per-layer coefficients

    During inference:
        KV_i = U @ c_i  (reconstruct on-the-fly)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

import geoopt
from geoopt.manifolds import Stiefel

from .config import MLAConfig
from .utils import orthonormalize_columns, check_orthonormality, random_orthonormal_columns


class XKVCompression(nn.Module):
    """Cross-layer KV compression module for a single layer within a group.

    This module handles compression/decompression for one layer, but shares
    its decompression matrices (W_uk, W_uv) with other layers in the same group.

    The key difference from per-layer MLACompression:
    - W_down_k, W_down_v are per-layer (different for each layer)
    - W_uk, W_uv are SHARED across all layers in the group

    Interface matches MLACompression for drop-in compatibility:
    - compress(h) -> c
    - decompress_k(c) -> K
    - decompress_v(c) -> V
    - check_orthonormality()
    - init_from_weights()
    - store_original_weights()
    - compute_reconstruction_loss()
    """

    def __init__(
        self,
        config: MLAConfig,
        layer_idx: int,
        group_layers: List[int],
        d_latent: int,
    ):
        """Initialize xKV compression for a single layer.

        Args:
            config: MLA configuration
            layer_idx: Index of this layer
            group_layers: List of layer indices in this compression group
            d_latent: Latent dimension (per-layer)
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.group_layers = group_layers
        self.d_latent = d_latent
        self.d_model = config.d_model
        self.d_kv = config.d_kv

        # Per-layer compression matrices (project hidden states to latent space)
        self.W_down_k = nn.Linear(self.d_model, d_latent, bias=False)
        self.W_down_v = nn.Linear(self.d_model, d_latent, bias=False)

        # Shared decompression matrices (orthonormal columns)
        # These will be set to shared references during group initialization
        stiefel = Stiefel(canonical=False)
        W_uk_init = random_orthonormal_columns(self.d_kv, d_latent).float()
        W_uv_init = random_orthonormal_columns(self.d_kv, d_latent).float()

        self.W_uk = geoopt.ManifoldParameter(W_uk_init, manifold=stiefel)
        self.W_uv = geoopt.ManifoldParameter(W_uv_init, manifold=stiefel)

        # Buffers for reconstruction loss (optional)
        self.register_buffer("W_k_original", None)
        self.register_buffer("W_v_original", None)

        # Buffers for K/V biases (required for models like Qwen that have biases)
        self.register_buffer("b_k", None)
        self.register_buffer("b_v", None)

    def compress(self, h: torch.Tensor) -> torch.Tensor:
        """Compress hidden states to latent representation.

        Args:
            h: Hidden states of shape (batch, seq_len, d_model)

        Returns:
            Latent representation of shape (batch, seq_len, 2 * d_latent)
            First d_latent dims are c_k, last d_latent dims are c_v
        """
        h_float = h.float()
        c_k = torch.nn.functional.linear(h_float, self.W_down_k.weight.float(), None)
        c_v = torch.nn.functional.linear(h_float, self.W_down_v.weight.float(), None)
        return torch.cat([c_k, c_v], dim=-1).to(h.dtype)

    def decompress_k(self, c: torch.Tensor) -> torch.Tensor:
        """Decompress latent representation to keys.

        Args:
            c: Latent representation of shape (batch, seq_len, 2 * d_latent)

        Returns:
            Keys of shape (batch, seq_len, d_kv)
        """
        c_k = c[..., : self.d_latent].float()
        k = torch.matmul(c_k, self.W_uk.float().T)
        # Add K bias if present (critical for models like Qwen)
        if self.b_k is not None:
            k = k + self.b_k.to(k.device).float()
        return k.to(c.dtype)

    def decompress_v(self, c: torch.Tensor) -> torch.Tensor:
        """Decompress latent representation to values.

        Args:
            c: Latent representation of shape (batch, seq_len, 2 * d_latent)

        Returns:
            Values of shape (batch, seq_len, d_kv)
        """
        c_v = c[..., self.d_latent :].float()
        v = torch.matmul(c_v, self.W_uv.float().T)
        # Add V bias if present (critical for models like Qwen)
        if self.b_v is not None:
            v = v + self.b_v.to(v.device).float()
        return v.to(c.dtype)

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

    def decompress(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress latent representation to keys and values.

        Args:
            c: Latent representation of shape (batch, seq_len, 2 * d_latent)

        Returns:
            Tuple of (keys, values), each of shape (batch, seq_len, d_kv)
        """
        return self.decompress_k(c), self.decompress_v(c)

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress hidden states and decompress to K, V.

        Args:
            h: Hidden states of shape (batch, seq_len, d_model)

        Returns:
            Tuple of (c, K, V) where:
                - c: (batch, seq_len, 2 * d_latent)
                - K: (batch, seq_len, d_kv)
                - V: (batch, seq_len, d_kv)
        """
        c = self.compress(h)
        K = self.decompress_k(c)
        V = self.decompress_v(c)
        return c, K, V

    def check_orthonormality(self) -> Dict[str, Tuple[float, float]]:
        """Check orthonormality of decompression matrices.

        W_uk and W_uv should have orthonormal columns (W.T @ W = I).

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
        W_uv: torch.Tensor,
    ) -> None:
        """Initialize from pre-computed weights.

        Args:
            W_down_k: K compression weights of shape (d_latent, d_model)
            W_down_v: V compression weights of shape (d_latent, d_model)
            W_uk: K decompression weights of shape (d_kv, d_latent)
            W_uv: V decompression weights of shape (d_kv, d_latent)
        """
        target_dtype = self.W_down_k.weight.dtype
        self.W_down_k.weight.data = W_down_k.to(device=self.W_down_k.weight.device, dtype=target_dtype)
        self.W_down_v.weight.data = W_down_v.to(device=self.W_down_v.weight.device, dtype=target_dtype)
        # Orthonormalize in float32 for numerical precision, then store in param's dtype
        self.W_uk.data = orthonormalize_columns(W_uk.to(self.W_uk.device).float()).to(self.W_uk.dtype)
        self.W_uv.data = orthonormalize_columns(W_uv.to(self.W_uv.device).float()).to(self.W_uv.dtype)

    def store_original_weights(self, W_k: torch.Tensor, W_v: torch.Tensor) -> None:
        """Store original K/V projection weights for reconstruction loss.

        Args:
            W_k: Original key projection weights of shape (d_kv, d_model)
            W_v: Original value projection weights of shape (d_kv, d_model)
        """
        self.register_buffer("W_k_original", W_k.clone().detach())
        self.register_buffer("W_v_original", W_v.clone().detach())

    def has_original_weights(self) -> bool:
        """Check if original weights are stored."""
        return self.W_k_original is not None and self.W_v_original is not None

    def compute_reconstruction_loss(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute MSE loss between reconstructed and original K/V.

        Args:
            h: Hidden states of shape (batch, seq_len, d_model)

        Returns:
            Tuple of (k_loss, v_loss) - MSE losses for K and V reconstruction

        Raises:
            ValueError: If original weights have not been stored
        """
        if not self.has_original_weights():
            raise ValueError(
                "Original weights not stored. Call store_original_weights() first."
            )

        h_float = h.float()
        K_target = torch.matmul(h_float, self.W_k_original.float().T)
        V_target = torch.matmul(h_float, self.W_v_original.float().T)
        if self.b_k is not None:
            K_target = K_target + self.b_k.float()
        if self.b_v is not None:
            V_target = V_target + self.b_v.float()

        c, K_recon, V_recon = self(h)

        k_loss = F.mse_loss(K_recon.float(), K_target)
        v_loss = F.mse_loss(V_recon.float(), V_target)
        return k_loss, v_loss

    def get_euclidean_params(self) -> List[torch.Tensor]:
        """Get Euclidean (non-manifold) parameters for standard optimizer."""
        return [self.W_down_k.weight, self.W_down_v.weight]

    def get_manifold_params(self) -> List[geoopt.ManifoldParameter]:
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
            f"layer_idx={self.layer_idx}, "
            f"group={self.group_layers}, "
            f"d_model={self.d_model}, "
            f"d_latent={self.d_latent}, "
            f"d_kv={self.d_kv}"
        )


class XKVCompressionGroup(nn.Module):
    """Manages a group of layers that share decompression matrices.

    This is the key component of xKV compression. It creates XKVCompression
    modules for each layer in the group, with SHARED W_uk and W_uv matrices.

    Key differences from per-layer MLA:
    - W_down_k, W_down_v are per-layer (different for each layer)
    - W_uk, W_uv are SHARED across all layers in the group

    Benefits:
    - Training shared W_uk, W_uv is more efficient (fewer parameters)
    - Updates benefit multiple layers simultaneously
    - Cross-layer information is captured in the shared basis
    """

    def __init__(
        self,
        config: MLAConfig,
        layer_indices: List[int],
        d_latent: int,
    ):
        """Initialize compression group with shared decompression matrices.

        Args:
            config: MLA configuration
            layer_indices: List of layer indices in this group
            d_latent: Latent dimension (per-layer)
        """
        super().__init__()
        self.config = config
        self.layer_indices = layer_indices
        self.d_latent = d_latent
        self.d_kv = config.d_kv

        # Shared decompression matrices (orthonormal columns)
        stiefel = Stiefel(canonical=False)
        W_uk_init = random_orthonormal_columns(self.d_kv, d_latent).float()
        W_uv_init = random_orthonormal_columns(self.d_kv, d_latent).float()

        self.shared_W_uk = geoopt.ManifoldParameter(W_uk_init, manifold=stiefel)
        self.shared_W_uv = geoopt.ManifoldParameter(W_uv_init, manifold=stiefel)

        # Per-layer compression modules (share W_uk, W_uv references)
        self.layer_compressions = nn.ModuleDict()
        for layer_idx in layer_indices:
            comp = XKVCompression(config, layer_idx, layer_indices, d_latent)
            # Share the decompression matrices - assign the SAME tensor
            comp.W_uk = self.shared_W_uk
            comp.W_uv = self.shared_W_uv
            self.layer_compressions[str(layer_idx)] = comp

    def get_compression(self, layer_idx: int) -> XKVCompression:
        """Get compression module for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            XKVCompression module for the layer

        Raises:
            KeyError: If layer_idx is not in this group
        """
        return self.layer_compressions[str(layer_idx)]

    def init_shared_basis(self, W_uk: torch.Tensor, W_uv: torch.Tensor) -> None:
        """Initialize the shared decompression matrices.

        Args:
            W_uk: Shared K decompression matrix of shape (d_kv, d_latent)
            W_uv: Shared V decompression matrix of shape (d_kv, d_latent)
        """
        # Orthonormalize in float32 for precision, then store in param's dtype
        self.shared_W_uk.data = orthonormalize_columns(W_uk.clone().float()).to(self.shared_W_uk.dtype)
        self.shared_W_uv.data = orthonormalize_columns(W_uv.clone().float()).to(self.shared_W_uv.dtype)

    def check_orthonormality(self) -> Dict[str, Tuple[float, float]]:
        """Check orthonormality of shared decompression matrices.

        Returns:
            Dictionary with 'W_uk' and 'W_uv' keys, each containing
            (max_error, mean_error) from identity matrix
        """
        return {
            "W_uk": check_orthonormality(self.shared_W_uk, mode="columns"),
            "W_uv": check_orthonormality(self.shared_W_uv, mode="columns"),
        }

    def project_to_manifold(self) -> None:
        """Project shared matrices back to Stiefel manifold."""
        with torch.no_grad():
            self.shared_W_uk.data = orthonormalize_columns(self.shared_W_uk.data)
            self.shared_W_uv.data = orthonormalize_columns(self.shared_W_uv.data)

    def get_euclidean_params(self) -> List[torch.Tensor]:
        """Get all Euclidean parameters from all layers in this group."""
        params = []
        for comp in self.layer_compressions.values():
            params.extend(comp.get_euclidean_params())
        return params

    def get_manifold_params(self) -> List[geoopt.ManifoldParameter]:
        """Get shared manifold parameters (only once, not per-layer)."""
        return [self.shared_W_uk, self.shared_W_uv]

    def _apply(self, fn):
        """Override _apply to keep shared parameters in float32."""
        super()._apply(fn)
        if self.shared_W_uk.dtype != torch.float32:
            self.shared_W_uk.data = self.shared_W_uk.data.float()
        if self.shared_W_uv.dtype != torch.float32:
            self.shared_W_uv.data = self.shared_W_uv.data.float()
        return self

    def extra_repr(self) -> str:
        return (
            f"layers={self.layer_indices}, "
            f"d_latent={self.d_latent}, "
            f"d_kv={self.d_kv}"
        )
