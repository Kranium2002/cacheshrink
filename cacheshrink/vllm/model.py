"""vLLM model implementation for cacheshrink compressed KV cache models.

CacheShrinkForCausalLM is the top-level model class registered with vLLM.
It builds a decoder stack with CacheShrinkMLAAttention layers that store
compressed c_kv in vLLM's paged KV cache.

Architecture adaptation (MLP type, norm type) is driven by the base model's
``model_type`` from the HF config.
"""

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors

from .attention import CacheShrinkMLAAttention
from .config import VLLMMLAConfig
from .weight_loader import load_cacheshrink_weights


def _get_act_fn(hidden_act: str):
    """Get activation function by name."""
    if hidden_act in ("silu", "swiglu"):
        return nn.SiLU()
    elif hidden_act == "gelu":
        return nn.GELU()
    elif hidden_act in ("gelu_new", "gelu_fast"):
        return nn.GELU(approximate="tanh")
    elif hidden_act == "relu":
        return nn.ReLU()
    else:
        return nn.SiLU()


class CacheShrinkMLP(nn.Module):
    """MLP layer adapted from the base model architecture.

    For LLaMA/Mistral/Qwen: SwiGLU with gate_up_proj + down_proj.
    For GPT-2: standard GELU MLP with up_proj + down_proj.
    """

    def __init__(self, mla_config: VLLMMLAConfig):
        super().__init__()
        d_model = mla_config.d_model
        intermediate_size = mla_config.intermediate_size

        if mla_config.model_type in ("gpt2",):
            # GPT-2 style: up -> act -> down
            self.up_proj = ColumnParallelLinear(d_model, intermediate_size, bias=True)
            self.down_proj = RowParallelLinear(intermediate_size, d_model, bias=True)
            self.act_fn = _get_act_fn(mla_config.hidden_act)
            self.use_gate = False
        else:
            # LLaMA/Mistral/Qwen style: SwiGLU
            self.gate_up_proj = MergedColumnParallelLinear(
                d_model,
                [intermediate_size, intermediate_size],
                bias=False,
            )
            self.down_proj = RowParallelLinear(intermediate_size, d_model, bias=False)
            self.act_fn = _get_act_fn(mla_config.hidden_act)
            self.use_gate = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gate:
            gate_up, _ = self.gate_up_proj(x)
            gate, up = gate_up.chunk(2, dim=-1)
            x = self.act_fn(gate) * up
            x, _ = self.down_proj(x)
        else:
            x, _ = self.up_proj(x)
            x = self.act_fn(x)
            x, _ = self.down_proj(x)
        return x


class CacheShrinkDecoderLayer(nn.Module):
    """Single decoder layer with MLA attention and MLP."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        mla_config: VLLMMLAConfig,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Attention with compressed KV cache
        self.self_attn = CacheShrinkMLAAttention(
            vllm_config=vllm_config,
            mla_config=mla_config,
            layer_idx=layer_idx,
            prefix=f"{prefix}.self_attn",
        )

        # MLP
        self.mlp = CacheShrinkMLP(mla_config)

        # Layer norm — RMSNorm for LLaMA family, LayerNorm for GPT-2.
        # RMSNorm supports fused (hidden_states, residual) -> (normed, residual) calls.
        self._use_fused_norm = False
        if mla_config.model_type in ("gpt2",):
            self.input_layernorm = nn.LayerNorm(mla_config.d_model, eps=mla_config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(
                mla_config.d_model, eps=mla_config.layer_norm_eps
            )
        else:
            try:
                from vllm.model_executor.layers.layernorm import RMSNorm

                self.input_layernorm = RMSNorm(mla_config.d_model, eps=mla_config.layer_norm_eps)
                self.post_attention_layernorm = RMSNorm(
                    mla_config.d_model, eps=mla_config.layer_norm_eps
                )
                self._use_fused_norm = True
            except ImportError:
                self.input_layernorm = nn.LayerNorm(
                    mla_config.d_model, eps=mla_config.layer_norm_eps
                )
                self.post_attention_layernorm = nn.LayerNorm(
                    mla_config.d_model, eps=mla_config.layer_norm_eps
                )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._use_fused_norm:
            # RMSNorm supports fused add+norm: (hidden, residual) -> (normed, new_residual)
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.self_attn(positions, hidden_states)

            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
        else:
            # nn.LayerNorm — manual residual handling
            if residual is None:
                residual = hidden_states
            else:
                hidden_states = hidden_states + residual
                residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(positions, hidden_states)

            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class CacheShrinkModel(nn.Module):
    """Transformer model with MLA compressed KV cache layers."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        mla_config: VLLMMLAConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.mla_config = mla_config

        self.embed_tokens = VocabParallelEmbedding(mla_config.vocab_size, mla_config.d_model)

        self.layers = nn.ModuleList(
            [
                CacheShrinkDecoderLayer(
                    vllm_config=vllm_config,
                    mla_config=mla_config,
                    layer_idx=i,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(mla_config.n_layers)
            ]
        )

        # Final norm
        self._use_fused_norm = False
        if mla_config.model_type in ("gpt2",):
            self.norm = nn.LayerNorm(mla_config.d_model, eps=mla_config.layer_norm_eps)
        else:
            try:
                from vllm.model_executor.layers.layernorm import RMSNorm

                self.norm = RMSNorm(mla_config.d_model, eps=mla_config.layer_norm_eps)
                self._use_fused_norm = True
            except ImportError:
                self.norm = nn.LayerNorm(mla_config.d_model, eps=mla_config.layer_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # Final norm — RMSNorm supports fused (hidden, residual), nn.LayerNorm does not
        if self._use_fused_norm and residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            if residual is not None:
                hidden_states = hidden_states + residual
            hidden_states = self.norm(hidden_states)
        return hidden_states


class CacheShrinkForCausalLM(nn.Module):
    """Top-level model class for vLLM serving of cacheshrink MLA models.

    Registered with vLLM's ModelRegistry via the plugin entry point.
    vLLM instantiates this class when ``architectures`` in config.json
    contains ``"CacheShrinkForCausalLM"``.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        hf_config = vllm_config.model_config.hf_config
        self.mla_config = VLLMMLAConfig.from_hf_config(hf_config)

        self.model = CacheShrinkModel(
            vllm_config=vllm_config,
            mla_config=self.mla_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )

        self.lm_head = ParallelLMHead(
            self.mla_config.vocab_size,
            self.mla_config.d_model,
            bias=False,
        )

        self.logits_processor = LogitsProcessor(self.mla_config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        load_cacheshrink_weights(self, weights, self.mla_config)
        # Return all parameter names as loaded (weight loader handles all mapping)
        return set(name for name, _ in self.named_parameters())
