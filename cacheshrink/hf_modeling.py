"""HuggingFace AutoModel integration for cacheshrink MLA models.

Provides CacheShrinkModelForCausalLM which enables loading MLA-compressed models
via the standard HuggingFace AutoModelForCausalLM.from_pretrained() pattern:

    model = AutoModelForCausalLM.from_pretrained("./mla-model", trust_remote_code=True)

This class is referenced by the auto_map in config.json and delegates loading
to cacheshrink's existing load_mla_model() function.
"""

import torch
from transformers import PreTrainedModel, PretrainedConfig


class CacheShrinkModelForCausalLM(PreTrainedModel):
    """Wrapper class that enables AutoModelForCausalLM.from_pretrained() for MLA models.

    This class is never instantiated directly. Its from_pretrained() classmethod
    delegates to cacheshrink.load_mla_model() and returns the actual HF model
    (e.g., LlamaForCausalLM with MLA attention), not a wrapper.
    """

    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load an MLA-compressed model using cacheshrink.load_mla_model().

        Maps HuggingFace kwargs to load_mla_model() parameters and returns the
        raw inner model (not a wrapper).

        Args:
            pretrained_model_name_or_path: Path to saved MLA model directory.
            **kwargs: HuggingFace kwargs (device_map, torch_dtype, etc.)

        Returns:
            The loaded HF model with MLA attention (e.g., LlamaForCausalLM).
        """
        from .saving import load_mla_model

        # Extract kwargs we can map to load_mla_model()
        config = kwargs.pop("config", None)
        device_map = kwargs.pop("device_map", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", True)

        # HF's auto dispatch consumes torch_dtype into the config object before
        # calling our from_pretrained. Recover it from config if not in kwargs.
        if torch_dtype is None and config is not None:
            config_dtype = getattr(config, "torch_dtype", None)
            if isinstance(config_dtype, torch.dtype):
                torch_dtype = config_dtype

        # Determine device from device_map
        device = None
        if device_map == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(device_map, str) and device_map not in ("auto", "balanced", "sequential"):
            device = device_map
        elif isinstance(device_map, torch.device):
            device = str(device_map)

        # Map torch_dtype
        dtype = None
        if torch_dtype is not None and torch_dtype != "auto":
            dtype = torch_dtype

        # Strip remaining HF-specific kwargs that load_mla_model doesn't accept
        for key in list(kwargs.keys()):
            if key in (
                "trust_remote_code",
                "cache_dir",
                "revision",
                "attn_implementation",
                "use_flash_attention_2",
                "token",
                "use_auth_token",
                "force_download",
                "proxies",
                "resume_download",
                "local_files_only",
                "use_safetensors",
                "output_loading_info",
                "quantization_config",
                "_fast_init",
                "subfolder",
                "variant",
            ):
                kwargs.pop(key)

        # Pop any remaining unknown kwargs to avoid TypeError
        kwargs.clear()

        model, _tokenizer = load_mla_model(
            pretrained_model_name_or_path,
            device=device,
            dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        return model
