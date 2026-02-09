"""Extract MLA configuration from HuggingFace config for vLLM.

VLLMMLAConfig is a dataclass that extracts all MLA-related parameters
from the HF config.json (specifically the ``cacheshrink_mla`` dict embedded
by ``save_for_vllm``). It also provides helper methods to convert back to
cacheshrink's native MLAConfig (needed to construct compression modules).
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import MLAConfig


@dataclass
class VLLMMLAConfig:
    """MLA configuration extracted from HF config for vLLM serving.

    Attributes:
        base_architectures: Original model architectures (e.g. ["LlamaForCausalLM"])
        model_type: HF model type (e.g. "llama", "mistral", "gpt2")
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_kv_heads: Original number of KV heads (before compression)
        d_head: Per-head dimension
        n_layers: Number of transformer layers
        vocab_size: Vocabulary size
        compression_method: "separate" or "xkv"
        d_latent: Latent dimension for compressed KV
        use_bias: Whether attention uses bias
        d_rope: RoPE dimension for decoupled_rope method
        uses_rope: Whether model uses rotary embeddings
        rope_theta: RoPE base frequency
        rope_scaling: RoPE scaling configuration
        use_cross_layer: Whether xKV cross-layer compression is used
        cross_layer_group_size: Number of layers per xKV group
        xkv_skip_early_layers: Number of early layers skipped from xKV
        intermediate_size: MLP intermediate dimension
        hidden_act: Activation function name
        layer_norm_eps: Layer norm epsilon
        max_position_embeddings: Maximum sequence length
        compression_ratio: Target compression ratio
    """

    # Base model info
    base_architectures: list[str]
    model_type: str
    d_model: int
    n_heads: int
    n_kv_heads: int  # original, before compression
    d_head: int
    n_layers: int
    vocab_size: int

    # MLA compression
    compression_method: str
    d_latent: int
    use_bias: bool
    d_rope: int = 64
    compression_ratio: float = 4.0

    # RoPE
    uses_rope: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict[str, Any]] = None

    # xKV cross-layer
    use_cross_layer: bool = False
    cross_layer_group_size: int = 4
    xkv_skip_early_layers: int = 0

    # Architecture details
    intermediate_size: int = 0
    hidden_act: str = "silu"
    layer_norm_eps: float = 1e-5
    max_position_embeddings: int = 2048

    # Extra config from original MLAConfig
    extra_config: dict[str, Any] = field(default_factory=dict)

    @property
    def n_compressed_kv_heads(self) -> int:
        """Number of virtual KV heads after compression.

        vLLM allocates KV cache as (num_blocks, block_size, num_kv_heads, head_size).
        By setting num_kv_heads = d_latent // d_head, we get smaller cache blocks.
        """
        return self.d_latent // self.d_head

    @property
    def d_kv(self) -> int:
        """Original total KV dimension: n_kv_heads * d_head."""
        return self.n_kv_heads * self.d_head

    def is_mla_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses MLA compression."""
        return True  # In vLLM mode, all layers use MLA

    def is_xkv_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses xKV cross-layer compression."""
        if not self.use_cross_layer:
            return False
        return layer_idx >= self.xkv_skip_early_layers

    def get_xkv_group(self, layer_idx: int) -> int:
        """Get which xKV group a layer belongs to.

        Returns -1 if the layer is not in an xKV group.
        """
        if not self.use_cross_layer:
            return -1
        if layer_idx < self.xkv_skip_early_layers:
            return -1
        adjusted_idx = layer_idx - self.xkv_skip_early_layers
        return adjusted_idx // self.cross_layer_group_size

    @property
    def n_groups(self) -> int:
        """Number of xKV compression groups."""
        if not self.use_cross_layer:
            return 0
        n_xkv_layers = max(0, self.n_layers - self.xkv_skip_early_layers)
        return (n_xkv_layers + self.cross_layer_group_size - 1) // self.cross_layer_group_size

    def get_group_layers(self, group_idx: int) -> list[int]:
        """Get layer indices for an xKV compression group."""
        if not self.use_cross_layer:
            return []
        start = group_idx * self.cross_layer_group_size + self.xkv_skip_early_layers
        end = min(start + self.cross_layer_group_size, self.n_layers)
        return list(range(start, end))

    def to_mla_config(self) -> MLAConfig:
        """Convert to cacheshrink's native MLAConfig.

        Needed for constructing MLACompression / XKVCompression modules
        which expect an MLAConfig instance.
        """
        return MLAConfig(
            model_name=f"vllm-{self.model_type}",
            model_type=self.model_type if self.model_type != "generic" else "generic",
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            d_model=self.d_model,
            d_head=self.d_head,
            n_layers=self.n_layers,
            compression_ratio=self.compression_ratio,
            d_latent=self.d_latent,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            vocab_size=self.vocab_size,
            use_bias=self.use_bias,
            layer_norm_eps=self.layer_norm_eps,
            compression_method=self.compression_method,
            d_rope=self.d_rope,
            use_cross_layer=self.use_cross_layer,
            cross_layer_group_size=self.cross_layer_group_size,
            xkv_skip_early_layers=self.xkv_skip_early_layers,
            keep_early_layers_original=False,
            extra_config={"uses_rope": self.uses_rope, **self.extra_config},
        )

    @classmethod
    def from_hf_config(cls, hf_config) -> "VLLMMLAConfig":
        """Extract VLLMMLAConfig from a HuggingFace config object.

        The HF config must have a ``cacheshrink_mla`` dict (embedded by
        ``save_for_vllm``).

        Args:
            hf_config: HuggingFace PretrainedConfig object

        Returns:
            VLLMMLAConfig instance

        Raises:
            ValueError: If cacheshrink_mla section is missing
        """
        mla_dict = getattr(hf_config, "cacheshrink_mla", None)
        if mla_dict is None:
            raise ValueError(
                "HF config missing 'cacheshrink_mla' section. "
                "Was this model saved with save_for_vllm()?"
            )

        # Extract base architectures from the mla config or from hf config
        base_archs = mla_dict.get(
            "base_architectures",
            getattr(hf_config, "base_architectures", ["unknown"]),
        )

        extra_config = mla_dict.get("extra_config", {})

        return cls(
            base_architectures=base_archs,
            model_type=mla_dict.get("model_type", "generic"),
            d_model=mla_dict["d_model"],
            n_heads=mla_dict["n_heads"],
            n_kv_heads=mla_dict.get("original_num_kv_heads", mla_dict["n_kv_heads"]),
            d_head=mla_dict["d_head"],
            n_layers=mla_dict["n_layers"],
            vocab_size=mla_dict.get("vocab_size", getattr(hf_config, "vocab_size", 50257)),
            compression_method=mla_dict.get("compression_method", "separate"),
            d_latent=mla_dict.get("d_latent") or mla_dict.get("computed_d_latent"),
            use_bias=mla_dict.get("use_bias", False),
            d_rope=mla_dict.get("d_rope", 64),
            compression_ratio=mla_dict.get("compression_ratio", 4.0),
            uses_rope=extra_config.get("uses_rope", True),
            rope_theta=mla_dict.get("rope_theta", 10000.0),
            rope_scaling=mla_dict.get("rope_scaling"),
            use_cross_layer=mla_dict.get("use_cross_layer", False),
            cross_layer_group_size=mla_dict.get("cross_layer_group_size", 4),
            xkv_skip_early_layers=mla_dict.get("xkv_skip_early_layers", 0),
            intermediate_size=getattr(hf_config, "intermediate_size", 0),
            hidden_act=getattr(hf_config, "hidden_act", "silu"),
            layer_norm_eps=mla_dict.get("layer_norm_eps", 1e-5),
            max_position_embeddings=mla_dict.get("max_position_embeddings", 2048),
            extra_config=extra_config,
        )
