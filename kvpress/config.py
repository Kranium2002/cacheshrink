"""Configuration for Multi-Head Latent Attention (MLA)."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json


@dataclass
class MLAConfig:
    """Configuration for MLA conversion and compression.

    Attributes:
        model_name: HuggingFace model name or path
        model_type: Type of model (gpt2, llama, mistral, qwen)
        n_heads: Number of attention heads
        n_kv_heads: Number of KV heads (< n_heads for GQA)
        d_model: Model hidden dimension
        d_head: Per-head dimension
        n_layers: Number of transformer layers
        compression_ratio: Target compression ratio (4-16x typical)
        d_latent: Latent dimension (auto-computed if None)
        max_position_embeddings: Maximum sequence length
        rope_theta: RoPE base frequency (if applicable)
        rope_scaling: RoPE scaling config (if applicable)
        vocab_size: Vocabulary size
        use_bias: Whether attention uses bias terms
        layer_norm_eps: Layer norm epsilon
    """

    model_name: str
    model_type: str
    n_heads: int
    n_kv_heads: int
    d_model: int
    d_head: int
    n_layers: int
    compression_ratio: float = 4.0
    d_latent: Optional[int] = None
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    vocab_size: int = 50257
    use_bias: bool = True
    layer_norm_eps: float = 1e-5

    # Additional model-specific attributes
    extra_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def d_kv(self) -> int:
        """Total KV dimension: n_kv_heads × d_head."""
        return self.n_kv_heads * self.d_head

    @property
    def computed_d_latent(self) -> int:
        """Compute d_latent from compression ratio if not specified.

        With separate compression for K and V:
        - Cache stores c_k (d_latent) + c_v (d_latent) = 2*d_latent
        - Original stores K (d_kv) + V (d_kv) = 2*d_kv
        - Compression ratio = 2*d_kv / (2*d_latent) = d_kv / d_latent
        - So d_latent = d_kv / compression_ratio

        Note: compression_ratio must be >= 1. A value of 1 means no compression.
        """
        if self.d_latent is not None:
            return self.d_latent
        # Ensure d_latent doesn't exceed d_kv (would break orthonormal column constraint)
        return min(self.d_kv, max(1, int(self.d_kv / self.compression_ratio)))

    @property
    def is_gqa(self) -> bool:
        """Whether model uses Grouped Query Attention."""
        return self.n_kv_heads < self.n_heads

    @property
    def n_rep(self) -> int:
        """Number of times to repeat KV heads for GQA."""
        return self.n_heads // self.n_kv_heads

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        compression_ratio: float = 4.0,
        d_latent: Optional[int] = None,
        **overrides
    ) -> "MLAConfig":
        """Create MLAConfig from a HuggingFace model.

        Args:
            model_name: HuggingFace model name or path
            compression_ratio: Target compression ratio
            d_latent: Override latent dimension (auto-computed if None)
            **overrides: Additional config overrides

        Returns:
            MLAConfig instance
        """
        from transformers import AutoConfig

        # Load HF config
        hf_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=overrides.pop("trust_remote_code", True)
        )

        # Detect model type
        model_type = cls._detect_model_type(hf_config)

        # Extract common attributes
        config_dict = cls._extract_config(hf_config, model_type)
        config_dict["model_name"] = model_name
        config_dict["model_type"] = model_type
        config_dict["compression_ratio"] = compression_ratio
        config_dict["d_latent"] = d_latent

        # Apply overrides
        for key, value in overrides.items():
            if key in config_dict:
                config_dict[key] = value
            else:
                if "extra_config" not in config_dict:
                    config_dict["extra_config"] = {}
                config_dict["extra_config"][key] = value

        return cls(**config_dict)

    @staticmethod
    def _detect_model_type(hf_config) -> str:
        """Detect model type from HuggingFace config."""
        model_type = getattr(hf_config, "model_type", "").lower()

        if model_type in ("gpt2",):
            return "gpt2"
        elif model_type in ("llama",):
            return "llama"
        elif model_type in ("mistral",):
            return "mistral"
        elif model_type in ("qwen2", "qwen"):
            return "qwen"
        else:
            # Try to infer from architecture
            architectures = getattr(hf_config, "architectures", [])
            if architectures:
                arch = architectures[0].lower()
                if "gpt2" in arch:
                    return "gpt2"
                elif "llama" in arch:
                    return "llama"
                elif "mistral" in arch:
                    return "mistral"
                elif "qwen" in arch:
                    return "qwen"

            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Supported: gpt2, llama, mistral, qwen")

    @staticmethod
    def _extract_config(hf_config, model_type: str) -> Dict[str, Any]:
        """Extract configuration from HuggingFace config."""
        config = {}

        if model_type == "gpt2":
            config["n_heads"] = hf_config.n_head
            config["n_kv_heads"] = hf_config.n_head  # GPT-2 uses MHA
            config["d_model"] = hf_config.n_embd
            config["d_head"] = hf_config.n_embd // hf_config.n_head
            config["n_layers"] = hf_config.n_layer
            config["max_position_embeddings"] = hf_config.n_positions
            config["vocab_size"] = hf_config.vocab_size
            config["use_bias"] = True
            config["layer_norm_eps"] = hf_config.layer_norm_epsilon
            # GPT-2 uses learned position embeddings, not RoPE
            config["rope_theta"] = 10000.0
            config["rope_scaling"] = None

        elif model_type in ("llama", "mistral", "qwen"):
            config["n_heads"] = hf_config.num_attention_heads
            config["n_kv_heads"] = getattr(
                hf_config, "num_key_value_heads", hf_config.num_attention_heads
            )
            config["d_model"] = hf_config.hidden_size
            config["d_head"] = hf_config.hidden_size // hf_config.num_attention_heads
            config["n_layers"] = hf_config.num_hidden_layers
            config["max_position_embeddings"] = hf_config.max_position_embeddings
            config["vocab_size"] = hf_config.vocab_size
            config["use_bias"] = getattr(hf_config, "attention_bias", False)
            config["layer_norm_eps"] = hf_config.rms_norm_eps
            config["rope_theta"] = getattr(hf_config, "rope_theta", 10000.0)
            config["rope_scaling"] = getattr(hf_config, "rope_scaling", None)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MLAConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MLAConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        return (
            f"MLAConfig(\n"
            f"  model_name={self.model_name!r},\n"
            f"  model_type={self.model_type!r},\n"
            f"  n_heads={self.n_heads},\n"
            f"  n_kv_heads={self.n_kv_heads},\n"
            f"  d_model={self.d_model},\n"
            f"  d_head={self.d_head},\n"
            f"  d_kv={self.d_kv},\n"
            f"  compression_ratio={self.compression_ratio},\n"
            f"  d_latent={self.computed_d_latent},\n"
            f"  is_gqa={self.is_gqa},\n"
            f"  n_layers={self.n_layers}\n"
            f")"
        )
