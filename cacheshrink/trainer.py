"""MLATrainer with Riemannian optimization for Stiefel manifold parameters.

Supports two training modes:
1. Standard LM loss (default) - trains to minimize perplexity
2. Knowledge distillation - trains MLA to match original model's outputs
   (RECOMMENDED for MLA training to avoid drifting from initialization)
"""

import os
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import geoopt

from .config import MLAConfig
from .utils import check_orthonormality


@dataclass
class TrainingConfig:
    """Configuration for MLA training."""

    # Learning rates (lower defaults for distillation stability)
    euclidean_lr: float = 1e-5
    riemannian_lr: float = 1e-4

    # Training parameters
    num_epochs: int = 3
    batch_size: int = 4
    max_length: int = 512
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Optimizer settings
    euclidean_weight_decay: float = 0.01
    riemannian_weight_decay: float = 0.0
    warmup_steps: int = 100
    lr_scheduler_type: str = "linear"

    # Distillation settings (RECOMMENDED, mutually exclusive with reconstruction loss)
    use_distillation: bool = True  # Use knowledge distillation from original model
    distillation_temperature: float = 2.0  # Temperature for softmax in distillation
    distillation_alpha: float = 0.9  # Weight of distillation loss (1-alpha for LM loss)

    # Reconstruction loss settings (alternative to distillation, mutually exclusive)
    # No teacher model needed - uses stored original K/V weights instead
    use_reconstruction_loss: bool = False  # Use direct K/V reconstruction loss
    reconstruction_alpha: float = 0.3  # Weight of reconstruction loss (blended with LM loss)

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 0  # 0 = disabled, set to e.g. 1000 to save every 1000 steps
    save_total_limit: int = 2  # Keep only last N checkpoints
    output_dir: str = "./mla_checkpoints"

    # Orthonormality monitoring
    check_orthonormality_steps: int = 100
    project_to_manifold: bool = False  # Usually not needed with geoopt


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []

        for text in texts:
            if not text or not text.strip():
                continue
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            if encoding["input_ids"].shape[1] > 10:  # Skip very short sequences
                self.encodings.append({
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def collate_fn(batch, pad_token_id):
    """Collate function with padding."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        # Pad input_ids
        padded_ids = torch.cat([
            item["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ])
        input_ids.append(padded_ids)

        # Pad attention mask
        padded_mask = torch.cat([
            item["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        attention_mask.append(padded_mask)

        # Labels are input_ids shifted, with padding as -100
        label = padded_ids.clone()
        label[padded_mask == 0] = -100
        labels.append(label)

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


class MLATrainer:
    """Trainer for MLA models with Riemannian optimization.

    Uses separate optimizers:
    - AdamW for Euclidean parameters (W_down_k, W_down_v compression matrices)
    - RiemannianAdam for Stiefel manifold parameters (W_uk, W_uv decompression matrices)

    The decompression matrices are constrained to have orthonormal columns (Stiefel manifold).
    RiemannianAdam automatically maintains this constraint during optimization.

    Supports knowledge distillation training (RECOMMENDED) to match original model outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[TrainingConfig] = None,
        euclidean_lr: float = 1e-5,
        riemannian_lr: float = 1e-4,
        teacher_model: Optional[nn.Module] = None,
        use_distillation: bool = True,
        use_reconstruction_loss: bool = False,
        reconstruction_alpha: float = 0.3,
        save_steps: int = 0,
    ):
        """Initialize trainer.

        Args:
            model: MLA-converted model
            tokenizer: HuggingFace tokenizer
            config: Training configuration (uses defaults if None)
            euclidean_lr: Learning rate for Euclidean parameters
            riemannian_lr: Learning rate for manifold parameters
            teacher_model: Original (non-MLA) model for distillation. If None and
                use_distillation=True, will load from mla_config.model_name
            use_distillation: Whether to use knowledge distillation (RECOMMENDED)
            use_reconstruction_loss: Whether to use direct K/V reconstruction loss.
                Requires model to have original weights stored (store_original_weights=True
                during conversion).
            reconstruction_alpha: Weight of reconstruction loss (0.0-1.0)
            save_steps: Save checkpoint every N steps (0 = disabled)
        """
        self.model = model
        self.tokenizer = tokenizer

        if config is None:
            config = TrainingConfig(
                euclidean_lr=euclidean_lr,
                riemannian_lr=riemannian_lr,
                use_distillation=use_distillation,
                use_reconstruction_loss=use_reconstruction_loss,
                reconstruction_alpha=reconstruction_alpha,
                save_steps=save_steps,
            )
        self.config = config

        # Get MLA config from model
        if not hasattr(model, "mla_config"):
            raise ValueError("Model does not have mla_config attribute")
        self.mla_config = model.mla_config

        # Validate mutually exclusive loss options
        if self.config.use_distillation and self.config.use_reconstruction_loss:
            raise ValueError(
                "use_distillation and use_reconstruction_loss are mutually exclusive. "
                "Choose one training approach: knowledge distillation (requires teacher model) "
                "or reconstruction loss (requires stored original weights)."
            )

        # Setup device
        self.device = next(model.parameters()).device

        # Setup teacher model for distillation
        self.teacher_model = None
        if self.config.use_distillation:
            if teacher_model is not None:
                self.teacher_model = teacher_model
            else:
                self._load_teacher_model()

        # Freeze non-MLA parameters
        self._freeze_non_mla_params()

        # Setup optimizers
        self._setup_optimizers()

        # Validate reconstruction loss setup
        if self.config.use_reconstruction_loss:
            self._validate_reconstruction_loss_setup()

        # Training state
        self.global_step = 0
        self.training_stats = {
            "train_loss": [],
            "distillation_loss": [],
            "lm_loss": [],
            "reconstruction_loss": [],
            "orthonormality_errors": [],
            "learning_rates": [],
        }

    def _load_teacher_model(self):
        """Load the original model as teacher for distillation."""
        try:
            from transformers import AutoModelForCausalLM

            print(f"Loading teacher model: {self.mla_config.model_name}")
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.mla_config.model_name,
                torch_dtype=next(self.model.parameters()).dtype,
                device_map="auto",
            )
            self.teacher_model.eval()
            # Freeze all teacher parameters
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            print("Teacher model loaded for distillation")
        except Exception as e:
            print(f"Warning: Could not load teacher model: {e}")
            print("Falling back to standard LM loss (not recommended)")
            self.config.use_distillation = False

    def _freeze_non_mla_params(self):
        """Freeze all parameters except MLA compression modules."""
        for name, param in self.model.named_parameters():
            if "mla_compression" in name or "mla." in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Also enable gradients for xKV group parameters if present
        if hasattr(self.model, "xkv_groups"):
            for group in self.model.xkv_groups.values():
                for param in group.parameters():
                    param.requires_grad = True

        # Count trainable parameters (xKV groups are already part of model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _setup_optimizers(self):
        """Setup separate optimizers for Euclidean and manifold parameters.

        For xKV models, the shared manifold parameters (shared_W_uk, shared_W_uv)
        are collected from xKV groups instead of individual layers.
        """
        self.euclidean_params = []
        self.manifold_params = []

        # Track which parameters we've already added (for xKV shared params)
        added_param_ids = set()

        # Check if model uses xKV compression
        use_xkv = hasattr(self.model, "xkv_groups") and self.model.xkv_groups

        if use_xkv:
            # For xKV, get params from compression groups (shared params only once)
            for group in self.model.xkv_groups.values():
                # Shared manifold params (only once per group)
                for param in group.get_manifold_params():
                    if id(param) not in added_param_ids:
                        self.manifold_params.append(param)
                        added_param_ids.add(id(param))

                # Per-layer Euclidean params
                for param in group.get_euclidean_params():
                    if id(param) not in added_param_ids:
                        self.euclidean_params.append(param)
                        added_param_ids.add(id(param))
        else:
            # Standard per-layer MLA
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                if "W_down" in name:
                    self.euclidean_params.append(param)
                elif "W_uk" in name or "W_uv" in name:
                    # Ensure param is a ManifoldParameter on Stiefel manifold
                    # This is needed because loading from disk loses ManifoldParameter type
                    if not isinstance(param, geoopt.ManifoldParameter):
                        # Convert to ManifoldParameter
                        stiefel = geoopt.manifolds.Stiefel()
                        manifold_param = geoopt.ManifoldParameter(param.data, manifold=stiefel)
                        manifold_param.requires_grad = True
                        # Replace in model
                        parts = name.split(".")
                        obj = self.model
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        setattr(obj, parts[-1], manifold_param)
                        self.manifold_params.append(manifold_param)
                    else:
                        self.manifold_params.append(param)

        print(f"Euclidean parameters: {len(self.euclidean_params)}")
        print(f"Manifold parameters: {len(self.manifold_params)}")
        if use_xkv:
            print(f"  (xKV: shared manifold params across {len(self.model.xkv_groups)} groups)")

        # AdamW for Euclidean parameters (W_down compression matrices)
        self.euclidean_optimizer = torch.optim.AdamW(
            self.euclidean_params,
            lr=self.config.euclidean_lr,
            weight_decay=self.config.euclidean_weight_decay,
        )

        # RiemannianAdam for Stiefel manifold parameters (W_uk, W_uv)
        # This optimizer respects the manifold geometry and maintains orthonormality
        self.riemannian_optimizer = geoopt.optim.RiemannianAdam(
            self.manifold_params,
            lr=self.config.riemannian_lr,
            weight_decay=self.config.riemannian_weight_decay,
        )

    def _get_attention_module_for_layer(self, layer_idx: int) -> Optional[nn.Module]:
        """Get the attention module for a specific layer.

        Uses mla_handler if available, falls back to hardcoded paths.
        """
        if hasattr(self.model, "mla_handler") and self.model.mla_handler is not None:
            handler = self.model.mla_handler
            layer = handler.get_layer_module(layer_idx)
            return getattr(layer, handler.get_attention_attribute_name())
        elif self.mla_config.model_type == "gpt2":
            return self.model.transformer.h[layer_idx].attn
        else:
            return self.model.model.layers[layer_idx].self_attn

    def _get_orthonormality_errors(self) -> Dict[str, float]:
        """Check orthonormality across all layers."""
        errors = {"W_uk_max": 0.0, "W_uv_max": 0.0, "W_uk_mean": 0.0, "W_uv_mean": 0.0}
        n_layers = 0

        for layer_idx in range(self.mla_config.n_layers):
            attn = self._get_attention_module_for_layer(layer_idx)

            # Get MLA compression module
            if hasattr(attn, "mla"):
                compression = attn.mla.mla_compression
            elif hasattr(attn, "mla_compression"):
                compression = attn.mla_compression
            else:
                continue

            # Check orthonormality
            layer_errors = compression.check_orthonormality()
            errors["W_uk_max"] = max(errors["W_uk_max"], layer_errors["W_uk"][0])
            errors["W_uv_max"] = max(errors["W_uv_max"], layer_errors["W_uv"][0])
            errors["W_uk_mean"] += layer_errors["W_uk"][1]
            errors["W_uv_mean"] += layer_errors["W_uv"][1]
            n_layers += 1

        if n_layers > 0:
            errors["W_uk_mean"] /= n_layers
            errors["W_uv_mean"] /= n_layers

        return errors

    def _validate_reconstruction_loss_setup(self) -> None:
        """Validate that model has original weights stored for reconstruction loss.

        Checks all layers to ensure consistent configuration across the model.
        """
        found_compression = False
        for layer_idx in range(self.mla_config.n_layers):
            compression = self._get_compression_module(layer_idx)
            if compression is None:
                continue
            found_compression = True
            if not compression.has_original_weights():
                raise ValueError(
                    f"Reconstruction loss requires original weights to be stored for all "
                    f"compression modules, but layer {layer_idx} is missing them. "
                    "Convert model with store_original_weights=True."
                )
        if not found_compression:
            raise ValueError("Could not find any compression module in model")

    def _get_compression_module(self, layer_idx: int):
        """Get the compression module for a specific layer.

        Args:
            layer_idx: Index of the layer

        Returns:
            The compression module, or None if not found
        """
        attn = self._get_attention_module_for_layer(layer_idx)

        if hasattr(attn, "mla"):
            return attn.mla.mla_compression
        elif hasattr(attn, "mla_compression"):
            return attn.mla_compression
        return None

    def _register_hidden_state_hooks(self) -> tuple:
        """Register hooks to capture hidden states during forward pass.

        Returns:
            Tuple of (hidden_states_dict, hooks_list)
            - hidden_states_dict will be populated during forward pass
            - hooks_list contains handles to remove after use
        """
        hidden_states_dict = {}
        hooks = []

        def make_hook(layer_idx):
            def hook(module, args, kwargs=None):
                # The first argument to attention is hidden_states
                if len(args) > 0:
                    h = args[0]
                else:
                    h = kwargs.get('hidden_states', None) if kwargs else None
                if h is not None:
                    # Keep gradient connection for reconstruction loss
                    hidden_states_dict[layer_idx] = h
            return hook

        # Register hooks on each attention module
        for layer_idx in range(self.mla_config.n_layers):
            attn = self._get_attention_module_for_layer(layer_idx)

            hook = attn.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
            hooks.append(hook)

        return hidden_states_dict, hooks

    def _remove_hooks(self, hooks: list) -> None:
        """Remove registered hooks."""
        for hook in hooks:
            hook.remove()

    def _compute_reconstruction_loss(self, hidden_states_dict: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Compute average reconstruction loss across all layers.

        Args:
            hidden_states_dict: Dictionary mapping layer_idx to hidden states
                (captured during forward pass via hooks)

        Returns:
            Average reconstruction loss (scalar tensor)
        """
        total_loss = 0.0
        n_layers = 0

        for layer_idx, h in hidden_states_dict.items():
            compression = self._get_compression_module(layer_idx)
            if compression is not None and compression.has_original_weights():
                k_loss, v_loss = compression.compute_reconstruction_loss(h)
                total_loss = total_loss + k_loss + v_loss
                n_layers += 1

        if n_layers > 0:
            return total_loss / (2 * n_layers)
        return torch.tensor(0.0, device=self.device)

    def train(
        self,
        dataset,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train the MLA model.

        Args:
            dataset: HuggingFace dataset or list of texts
            num_epochs: Override config num_epochs
            batch_size: Override config batch_size
            max_length: Override config max_length

        Returns:
            Training statistics
        """
        num_epochs = num_epochs or self.config.num_epochs
        batch_size = batch_size or self.config.batch_size
        max_length = max_length or self.config.max_length

        # Prepare dataset
        if hasattr(dataset, "__iter__") and not isinstance(dataset, (str, list)):
            # HuggingFace dataset
            text_key = "text" if "text" in dataset.column_names else dataset.column_names[0]
            texts = [item[text_key] for item in dataset if item[text_key]]
        else:
            texts = list(dataset)

        train_dataset = TextDataset(texts, self.tokenizer, max_length)
        print(f"Training on {len(train_dataset)} sequences")

        # Create dataloader
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, pad_token_id),
        )

        # Training loop
        self.model.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()

        total_steps = len(dataloader) * num_epochs

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_distill_loss = 0.0
            epoch_lm_loss = 0.0
            epoch_recon_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch in progress_bar:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Register hooks before forward pass if reconstruction loss is enabled
                hidden_states_dict = None
                hooks = None
                if self.config.use_reconstruction_loss:
                    hidden_states_dict, hooks = self._register_hidden_state_hooks()

                # Forward pass (captures hidden states if hooks are registered)
                outputs = self.model(**batch)

                # Remove hooks immediately after forward pass
                if hooks is not None:
                    self._remove_hooks(hooks)

                # Compute loss
                if self.config.use_distillation and self.teacher_model is not None:
                    # Get teacher logits (no grad)
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                        teacher_logits = teacher_outputs.logits

                    # Student logits
                    student_logits = outputs.logits

                    # Distillation loss: KL divergence with temperature
                    T = self.config.distillation_temperature
                    distill_loss = F.kl_div(
                        F.log_softmax(student_logits / T, dim=-1),
                        F.softmax(teacher_logits / T, dim=-1),
                        reduction="batchmean",
                    ) * (T * T)  # Scale by T^2 as per Hinton et al.

                    # Combine with LM loss
                    alpha = self.config.distillation_alpha
                    lm_loss = outputs.loss
                    loss = alpha * distill_loss + (1 - alpha) * lm_loss

                    epoch_distill_loss += distill_loss.item()
                    epoch_lm_loss += lm_loss.item()
                else:
                    loss = outputs.loss
                    epoch_lm_loss += loss.item()

                # Add reconstruction loss if enabled (uses hidden states captured during forward pass)
                if self.config.use_reconstruction_loss and hidden_states_dict is not None:
                    recon_loss = self._compute_reconstruction_loss(hidden_states_dict)
                    epoch_recon_loss += recon_loss.item()

                    # Blend reconstruction loss with LM loss
                    alpha = self.config.reconstruction_alpha
                    loss = (1 - alpha) * loss + alpha * recon_loss

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients for Euclidean params only
                    # Don't clip manifold params - let RiemannianAdam handle them properly
                    if self.config.max_grad_norm > 0 and self.euclidean_params:
                        torch.nn.utils.clip_grad_norm_(
                            self.euclidean_params,
                            self.config.max_grad_norm,
                        )

                    # Step optimizers
                    self.euclidean_optimizer.step()
                    self.riemannian_optimizer.step()

                    # Zero gradients
                    self.euclidean_optimizer.zero_grad()
                    self.riemannian_optimizer.zero_grad()

                # Update stats
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    self.training_stats["train_loss"].append(avg_loss)

                    postfix = {"loss": f"{avg_loss:.4f}"}

                    if self.config.use_distillation:
                        avg_distill = epoch_distill_loss / num_batches
                        avg_lm = epoch_lm_loss / num_batches
                        self.training_stats["distillation_loss"].append(avg_distill)
                        self.training_stats["lm_loss"].append(avg_lm)
                        postfix["distill"] = f"{avg_distill:.4f}"

                    if self.config.use_reconstruction_loss:
                        avg_recon = epoch_recon_loss / num_batches
                        self.training_stats["reconstruction_loss"].append(avg_recon)
                        postfix["recon"] = f"{avg_recon:.4f}"

                    progress_bar.set_postfix(postfix)

                # Check orthonormality
                if self.global_step % self.config.check_orthonormality_steps == 0:
                    errors = self._get_orthonormality_errors()
                    self.training_stats["orthonormality_errors"].append(errors)

                    if errors["W_uk_max"] > 1e-3 or errors["W_uv_max"] > 1e-3:
                        print(f"\nOrthonormality drift: W_uk={errors['W_uk_max']:.2e}, "
                              f"W_uv={errors['W_uv_max']:.2e}")

                # Save checkpoint
                if self.config.save_steps > 0 and self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()

            # End of epoch stats
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

        # Final orthonormality check
        final_errors = self._get_orthonormality_errors()
        print(f"\nFinal orthonormality errors:")
        print(f"  W_uk max: {final_errors['W_uk_max']:.2e}")
        print(f"  W_uv max: {final_errors['W_uv_max']:.2e}")

        return self.training_stats

    def _save_checkpoint(self):
        """Save a training checkpoint."""
        checkpoint_dir = os.path.join(
            self.config.output_dir,
            f"checkpoint-{self.global_step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model state
        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_dir, "pytorch_model.bin")
        )

        # Save optimizer states
        torch.save(
            {
                "euclidean_optimizer": self.euclidean_optimizer.state_dict(),
                "riemannian_optimizer": self.riemannian_optimizer.state_dict(),
                "global_step": self.global_step,
            },
            os.path.join(checkpoint_dir, "optimizer.bin")
        )

        print(f"Saved checkpoint to {checkpoint_dir}")

    def cleanup(self):
        """Release optimizer states and internal references to free GPU memory.

        Call this before deleting the trainer to ensure all CUDA tensors are freed.
        """
        # Clear optimizer states (momentum + variance buffers on GPU)
        if hasattr(self, "euclidean_optimizer"):
            self.euclidean_optimizer.state.clear()
        if hasattr(self, "riemannian_optimizer"):
            self.riemannian_optimizer.state.clear()

        # Drop parameter lists
        self.euclidean_params = []
        self.manifold_params = []

        # Drop teacher model
        if self.teacher_model is not None:
            del self.teacher_model
            self.teacher_model = None

        # Drop model reference (caller may still hold their own ref)
        self.model = None

    def load_checkpoint(self, checkpoint_dir: str):
        """Load a training checkpoint."""
        # Load model state
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Load optimizer states
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.bin")
        if os.path.exists(optimizer_path):
            checkpoint = torch.load(optimizer_path, map_location=self.device)
            self.euclidean_optimizer.load_state_dict(checkpoint["euclidean_optimizer"])
            self.riemannian_optimizer.load_state_dict(checkpoint["riemannian_optimizer"])
            self.global_step = checkpoint["global_step"]

        print(f"Loaded checkpoint from {checkpoint_dir}")
