"""MLA Attention module with compressed KV cache."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MLAConfig
from .compression import MLACompression
from .utils import repeat_kv


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation.

    This implementation supports both standard RoPE and various scaling methods
    used by LLaMA, Mistral, etc.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        rope_type: str = "default",
        device: Optional[torch.device] = None,
    ):
        """Initialize RoPE.

        Args:
            dim: Rotary dimension (usually d_head)
            max_position_embeddings: Maximum sequence length
            base: Base frequency (theta)
            scaling_factor: Scaling factor for position indices
            rope_type: Type of RoPE ("default", "linear", "dynamic")
            device: Device for computation
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.rope_type = rope_type

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings, device)

    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
    ):
        """Build or extend the cos/sin cache."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        if self.scaling_factor != 1.0:
            t = t / self.scaling_factor

        # Compute frequencies
        freqs = torch.outer(t, self.inv_freq.to(device))
        # Concatenate for both sin and cos
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin embeddings for positions.

        Args:
            x: Input tensor (used for dtype)
            position_ids: Position indices of shape (batch, seq_len)
            seq_len: Sequence length (used if position_ids not provided)

        Returns:
            Tuple of (cos, sin) each of shape (seq_len, dim) or (batch, seq_len, dim)
        """
        if seq_len is None:
            seq_len = x.shape[2] if x.dim() == 4 else x.shape[1]

        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device)

        if position_ids is not None:
            # Gather cos/sin for specific positions
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
        else:
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K.

    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim)
        cos: Cosine embeddings
        sin: Sine embeddings
        position_ids: Position indices (optional)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting

    Returns:
        Tuple of (q_embed, k_embed) with rotary embeddings applied
    """
    # Reshape cos/sin for broadcasting
    # From (seq_len, head_dim) to (1, 1, seq_len, head_dim)
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        # (batch, seq_len, head_dim) -> (batch, 1, seq_len, head_dim)
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MLAAttention(nn.Module):
    """Multi-Head Latent Attention with compressed KV cache.

    This is a drop-in replacement for standard multi-head attention that:
    1. Projects Q normally
    2. Compresses hidden states to latent KV representation
    3. Decompresses to K, V on the fly
    4. Stores only c_kv in the KV cache (d_latent instead of 2*d_kv)
    5. Supports both MHA and GQA

    Supports three compression methods:
    - "separate": Separate K/V compression (default, best quality)
    - "joint": Joint K/V with shared latent (2x more compression)
    - "decoupled_rope": Decoupled RoPE (preserves positional info)

    RoPE is applied after decompression to match original model behavior.
    """

    def __init__(
        self,
        config: MLAConfig,
        layer_idx: int = 0,
        compression_method: str = "separate",
        d_rope: int = 64,
    ):
        """Initialize MLA attention.

        Args:
            config: MLA configuration
            layer_idx: Layer index (for cache)
            compression_method: "separate", "joint", or "decoupled_rope"
            d_rope: RoPE dimension for decoupled_rope method
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compression_method = compression_method

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_rep = config.n_rep

        # Query projection (standard)
        self.q_proj = nn.Linear(
            config.d_model,
            config.n_heads * config.d_head,
            bias=config.use_bias,
        )

        # MLA compression module - depends on method
        if compression_method == "separate":
            self.mla_compression = MLACompression(config)
        elif compression_method == "joint":
            from .improved_compression import JointKVCompression
            self.mla_compression = JointKVCompression(config)
        elif compression_method == "decoupled_rope":
            from .improved_compression import DecoupledRoPECompression
            self.mla_compression = DecoupledRoPECompression(
                d_model=config.d_model,
                d_kv=config.d_kv,
                d_latent=config.computed_d_latent,
                d_rope=d_rope,
            )
        else:
            raise ValueError(f"Unknown compression_method: {compression_method}")

        # Output projection
        self.o_proj = nn.Linear(
            config.n_heads * config.d_head,
            config.d_model,
            bias=config.use_bias,
        )

        # RoPE (for models that use it)
        if config.model_type != "gpt2":
            self.rotary_emb = RotaryEmbedding(
                dim=config.d_head,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            self.rotary_emb = None

        # Attention scaling
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            hidden_states: Input of shape (batch, seq_len, d_model)
            attention_mask: Attention mask
            position_ids: Position IDs for RoPE
            past_key_value: Cached (c_kv,) from previous steps
            output_attentions: Whether to return attention weights
            use_cache: Whether to return updated cache

        Returns:
            Tuple of:
                - output: (batch, seq_len, d_model)
                - past_key_value: Updated cache (c_kv,) if use_cache
                - attention_weights: If output_attentions
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Query projection
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(
            batch_size, seq_len, self.n_heads, self.d_head
        ).transpose(1, 2)

        # Compress to latent space
        c_kv = self.mla_compression.compress(hidden_states)

        # Handle KV cache
        if past_key_value is not None:
            # Concatenate with past latent states
            past_c_kv = past_key_value[0]
            c_kv = torch.cat([past_c_kv, c_kv], dim=1)

        # Create new cache
        if use_cache:
            new_past_key_value = (c_kv,)
        else:
            new_past_key_value = None

        # Decompress to K, V
        key_states = self.mla_compression.decompress_k(c_kv)
        value_states = self.mla_compression.decompress_v(c_kv)

        # Reshape K, V for attention
        # key_states: (batch, kv_seq_len, d_kv) -> (batch, n_kv_heads, kv_seq_len, d_head)
        kv_seq_len = c_kv.shape[1]
        key_states = key_states.view(
            batch_size, kv_seq_len, self.n_kv_heads, self.d_head
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, kv_seq_len, self.n_kv_heads, self.d_head
        ).transpose(1, 2)

        # Apply RoPE if applicable
        if self.rotary_emb is not None:
            # Determine the maximum position we need
            past_len = kv_seq_len - seq_len

            if position_ids is None:
                # Create position IDs for the current tokens
                position_ids = torch.arange(
                    past_len, past_len + seq_len, device=hidden_states.device
                ).unsqueeze(0).expand(batch_size, -1)

            # Get the maximum position we need to handle
            max_pos = max(kv_seq_len, int(position_ids.max().item()) + 1)

            # Get cos/sin for all positions needed
            cos, sin = self.rotary_emb(value_states, seq_len=max_pos)

            # For query, we only need current positions
            # For key, we need all positions in the cache
            if cos.dim() == 2:
                # cos/sin are (max_pos, head_dim)
                # Clamp position_ids to valid range
                safe_position_ids = position_ids.clamp(0, cos.size(0) - 1)
                q_cos = cos[safe_position_ids]  # (batch, seq_len, head_dim)
                q_sin = sin[safe_position_ids]
                # For keys, need all positions
                k_positions = torch.arange(kv_seq_len, device=hidden_states.device)
                k_cos = cos[k_positions]  # (kv_seq_len, head_dim)
                k_sin = sin[k_positions]
            else:
                q_cos, q_sin = cos, sin
                k_cos, k_sin = cos, sin

            # Apply RoPE to Q (only current positions)
            query_states = (query_states * q_cos.unsqueeze(1)) + (
                rotate_half(query_states) * q_sin.unsqueeze(1)
            )
            # Apply RoPE to K (all positions)
            if k_cos.dim() == 2:
                k_cos = k_cos.unsqueeze(0).unsqueeze(0)
                k_sin = k_sin.unsqueeze(0).unsqueeze(0)
            key_states = (key_states * k_cos) + (rotate_half(key_states) * k_sin)

        # Repeat KV heads for GQA
        key_states = repeat_kv(key_states, self.n_rep)
        value_states = repeat_kv(value_states, self.n_rep)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale

        # Apply causal mask for autoregressive generation
        # Always apply causal masking - this is required for autoregressive models
        causal_mask = torch.triu(
            torch.full((seq_len, kv_seq_len), float("-inf"), device=hidden_states.device, dtype=attn_weights.dtype),
            diagonal=kv_seq_len - seq_len + 1,
        )
        attn_weights = attn_weights + causal_mask

        # Apply additional attention mask if provided
        if attention_mask is not None:
            # Handle different mask shapes
            if attention_mask.dim() == 2:
                # (batch, kv_seq_len) -> (batch, 1, 1, kv_seq_len)
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                # (batch, seq_len, kv_seq_len) -> (batch, 1, seq_len, kv_seq_len)
                attention_mask = attention_mask.unsqueeze(1)
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.d_head)

        # Output projection
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            return attn_output, new_past_key_value, attn_weights
        return attn_output, new_past_key_value, None

    def check_orthonormality(self):
        """Check orthonormality of compression matrices."""
        return self.mla_compression.check_orthonormality()


class MLACache:
    """Custom cache class for MLA attention.

    Stores only the compressed c_kv representations instead of full K, V tensors.
    """

    def __init__(self):
        self.cache: dict = {}

    def update(
        self,
        layer_idx: int,
        c_kv: torch.Tensor,
    ) -> torch.Tensor:
        """Update cache for a layer.

        Args:
            layer_idx: Layer index
            c_kv: New latent states to cache

        Returns:
            Full cached c_kv (past + new)
        """
        if layer_idx in self.cache:
            self.cache[layer_idx] = torch.cat([self.cache[layer_idx], c_kv], dim=1)
        else:
            self.cache[layer_idx] = c_kv
        return self.cache[layer_idx]

    def get(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get cached states for a layer."""
        return self.cache.get(layer_idx, None)

    def clear(self):
        """Clear all cached states."""
        self.cache.clear()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get current sequence length in cache."""
        if layer_idx in self.cache:
            return self.cache[layer_idx].shape[1]
        return 0

    def get_memory_usage(self) -> int:
        """Get total memory usage in bytes."""
        total = 0
        for c_kv in self.cache.values():
            total += c_kv.element_size() * c_kv.numel()
        return total
