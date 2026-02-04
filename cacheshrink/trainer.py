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

    # Distillation settings (RECOMMENDED)
    use_distillation: bool = True  # Use knowledge distillation from original model
    distillation_temperature: float = 2.0  # Temperature for softmax in distillation
    distillation_alpha: float = 0.9  # Weight of distillation loss (1-alpha for LM loss)

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
    - AdamW for Euclidean parameters (W_down)
    - RiemannianAdam for Stiefel manifold parameters (W_uk, W_uv)

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
            save_steps: Save checkpoint every N steps (0 = disabled)
        """
        self.model = model
        self.tokenizer = tokenizer

        if config is None:
            config = TrainingConfig(
                euclidean_lr=euclidean_lr,
                riemannian_lr=riemannian_lr,
                use_distillation=use_distillation,
                save_steps=save_steps,
            )
        self.config = config

        # Get MLA config from model
        if not hasattr(model, "mla_config"):
            raise ValueError("Model does not have mla_config attribute")
        self.mla_config = model.mla_config

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

        # Training state
        self.global_step = 0
        self.training_stats = {
            "train_loss": [],
            "distillation_loss": [],
            "lm_loss": [],
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

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _setup_optimizers(self):
        """Setup separate optimizers for Euclidean and manifold parameters."""
        euclidean_params = []
        manifold_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "W_down" in name:
                euclidean_params.append(param)
            elif "W_uk" in name or "W_uv" in name:
                manifold_params.append(param)

        print(f"Euclidean parameters: {len(euclidean_params)}")
        print(f"Manifold parameters: {len(manifold_params)}")

        # AdamW for Euclidean parameters
        self.euclidean_optimizer = torch.optim.AdamW(
            euclidean_params,
            lr=self.config.euclidean_lr,
            weight_decay=self.config.euclidean_weight_decay,
        )

        # RiemannianAdam for manifold parameters
        self.riemannian_optimizer = geoopt.optim.RiemannianAdam(
            manifold_params,
            lr=self.config.riemannian_lr,
            weight_decay=self.config.riemannian_weight_decay,
        )

    def _get_orthonormality_errors(self) -> Dict[str, float]:
        """Check orthonormality across all layers."""
        errors = {"W_uk_max": 0.0, "W_uv_max": 0.0, "W_uk_mean": 0.0, "W_uv_mean": 0.0}
        n_layers = 0

        for layer_idx in range(self.mla_config.n_layers):
            # Find attention module
            if self.mla_config.model_type == "gpt2":
                attn = self.model.transformer.h[layer_idx].attn
            else:
                attn = self.model.model.layers[layer_idx].self_attn

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
            num_batches = 0

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch in progress_bar:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)

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

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
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
                    if self.config.use_distillation:
                        avg_distill = epoch_distill_loss / num_batches
                        avg_lm = epoch_lm_loss / num_batches
                        self.training_stats["distillation_loss"].append(avg_distill)
                        self.training_stats["lm_loss"].append(avg_lm)
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "distill": f"{avg_distill:.4f}"
                        })
                    else:
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

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
