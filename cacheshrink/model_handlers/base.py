"""Base class for model-specific handlers."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Any
import torch
import torch.nn as nn

from ..config import MLAConfig


class ModelHandler(ABC):
    """Abstract base class for model-specific handling.

    Each model family (GPT-2, LLaMA, Mistral, Qwen) has different:
    - Weight layouts (combined vs separate K/V projections)
    - Module paths (transformer.h vs model.layers)
    - Attention class names
    - Position embedding types (learned vs RoPE)

    Handlers abstract these differences for the conversion pipeline.
    """

    def __init__(self, model: nn.Module, config: MLAConfig):
        """Initialize handler.

        Args:
            model: HuggingFace model
            config: MLA configuration
        """
        self.model = model
        self.config = config

    @abstractmethod
    def get_attention_module(self, layer_idx: int) -> nn.Module:
        """Get the attention module for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Attention module
        """
        pass

    @abstractmethod
    def get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get the full transformer layer module.

        Args:
            layer_idx: Layer index

        Returns:
            Layer module
        """
        pass

    @abstractmethod
    def extract_qkv_weights(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Extract Q, K, V projection weights from a layer.

        Args:
            layer_idx: Layer index

        Returns:
            Tuple of (W_q, W_k, W_v, W_o) where each is (out_features, in_features)
            W_o may be None if not needed
        """
        pass

    @abstractmethod
    def extract_qkv_biases(
        self, layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract Q, K, V, O projection biases from a layer.

        Args:
            layer_idx: Layer index

        Returns:
            Tuple of (b_q, b_k, b_v, b_o), each may be None if no bias
        """
        pass

    @abstractmethod
    def replace_attention(self, layer_idx: int, mla_attention: nn.Module) -> None:
        """Replace the original attention with MLA attention.

        Args:
            layer_idx: Layer index
            mla_attention: New MLA attention module
        """
        pass

    @abstractmethod
    def get_attention_attribute_name(self) -> str:
        """Get the attribute name for attention in the layer.

        Returns:
            Attribute name (e.g., "attn", "self_attn")
        """
        pass

    def get_num_layers(self) -> int:
        """Get number of transformer layers."""
        return self.config.n_layers

    def get_layer_prefix(self, layer_idx: int) -> str:
        """Get the state dict prefix for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            Prefix string for state dict keys
        """
        pass

    @abstractmethod
    def get_embed_tokens(self) -> nn.Module:
        """Get the token embedding module."""
        pass

    @abstractmethod
    def get_output_layer(self) -> nn.Module:
        """Get the output/LM head layer."""
        pass

    def freeze_non_mla_params(self) -> None:
        """Freeze all parameters except MLA compression modules."""
        for name, param in self.model.named_parameters():
            if "mla_compression" not in name:
                param.requires_grad = False

    def get_mla_params(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """Get MLA parameters separated by type.

        Returns:
            Tuple of (euclidean_params, manifold_params)
        """
        euclidean_params = []
        manifold_params = []

        for name, param in self.model.named_parameters():
            if "mla_compression" in name:
                if "W_down" in name:
                    euclidean_params.append(param)
                elif "W_uk" in name or "W_uv" in name:
                    manifold_params.append(param)

        return euclidean_params, manifold_params
