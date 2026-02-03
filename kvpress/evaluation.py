"""Evaluation functions for MLA models."""

import math
from typing import Optional, Dict, Any, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import MLAConfig
from .utils import format_size


def compute_perplexity(
    model: nn.Module,
    tokenizer,
    dataset,
    batch_size: int = 8,
    max_length: int = 512,
    stride: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> float:
    """Compute perplexity on a dataset.

    Uses a sliding window approach for sequences longer than max_length.

    Args:
        model: Language model
        tokenizer: Tokenizer
        dataset: HuggingFace dataset or list of texts
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        stride: Stride for sliding window (defaults to max_length // 2)
        device: Device for computation
        verbose: Whether to show progress bar

    Returns:
        Perplexity value
    """
    if device is None:
        device = next(model.parameters()).device

    if stride is None:
        stride = max_length // 2

    model.eval()

    # Extract texts from dataset
    if hasattr(dataset, "__iter__") and not isinstance(dataset, (str, list)):
        text_key = "text" if "text" in dataset.column_names else dataset.column_names[0]
        texts = [item[text_key] for item in dataset if item[text_key]]
    else:
        texts = list(dataset)

    total_loss = 0.0
    total_tokens = 0

    iterator = texts
    if verbose:
        iterator = tqdm(texts, desc="Computing perplexity")

    for text in iterator:
        if not text or len(text.strip()) < 10:
            continue

        # Tokenize
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            max_length=None,
        )
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.shape[1]

        if seq_len <= 1:
            continue

        # Sliding window evaluation
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # Tokens to predict

            input_chunk = input_ids[:, begin_loc:end_loc]

            # Create target labels (shift by 1)
            target_ids = input_chunk.clone()
            target_ids[:, :-trg_len] = -100  # Mask context tokens

            with torch.no_grad():
                outputs = model(input_chunk, labels=target_ids)
                # Loss is averaged over non-masked tokens
                neg_log_likelihood = outputs.loss

            # Accumulate loss weighted by number of tokens
            num_tokens = (target_ids != -100).sum().item()
            total_loss += neg_log_likelihood.item() * num_tokens
            total_tokens += num_tokens

            prev_end_loc = end_loc
            if end_loc >= seq_len:
                break

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    if verbose:
        print(f"Perplexity: {perplexity:.2f} (avg loss: {avg_loss:.4f}, tokens: {total_tokens:,})")

    return perplexity


def measure_cache_memory(
    model: nn.Module,
    sequence_lengths: Optional[List[int]] = None,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Measure KV cache memory usage.

    Compares MLA's compressed cache to standard KV cache.

    Args:
        model: MLA-converted model
        sequence_lengths: Sequence lengths to measure (defaults to [128, 512, 2048])
        batch_size: Batch size

    Returns:
        Dictionary with memory statistics
    """
    if sequence_lengths is None:
        sequence_lengths = [128, 512, 2048]

    if not hasattr(model, "mla_config"):
        raise ValueError("Model does not have mla_config attribute")

    config = model.mla_config

    # Calculate dimensions
    d_latent = config.computed_d_latent
    d_kv = config.d_kv
    n_layers = config.n_layers

    # Assume float16 (2 bytes per element)
    bytes_per_element = 2

    # MLA stores c_kv = [c_k, c_v], so cache size is 2*d_latent per token
    mla_cache_dim = 2 * d_latent

    results = {
        "config": {
            "d_latent": d_latent,
            "d_kv": d_kv,
            "n_layers": n_layers,
            "theoretical_compression_ratio": (2 * d_kv) / mla_cache_dim,
        },
        "per_sequence_length": {},
    }

    for seq_len in sequence_lengths:
        # MLA cache size: batch × seq_len × 2*d_latent × n_layers (stores both c_k and c_v)
        mla_cache_bytes = batch_size * seq_len * mla_cache_dim * n_layers * bytes_per_element

        # Standard KV cache size: batch × seq_len × d_kv × 2 (K and V) × n_layers
        standard_cache_bytes = batch_size * seq_len * d_kv * 2 * n_layers * bytes_per_element

        compression_ratio = standard_cache_bytes / mla_cache_bytes

        results["per_sequence_length"][seq_len] = {
            "mla_cache_bytes": mla_cache_bytes,
            "mla_cache_formatted": format_size(mla_cache_bytes),
            "standard_cache_bytes": standard_cache_bytes,
            "standard_cache_formatted": format_size(standard_cache_bytes),
            "compression_ratio": compression_ratio,
            "memory_saved_bytes": standard_cache_bytes - mla_cache_bytes,
            "memory_saved_formatted": format_size(standard_cache_bytes - mla_cache_bytes),
        }

    # Summary statistics
    results["compression_ratio"] = results["config"]["theoretical_compression_ratio"]

    return results


def compare_outputs(
    original_model: nn.Module,
    mla_model: nn.Module,
    tokenizer,
    texts: List[str],
    max_length: int = 128,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compare outputs between original and MLA model.

    Args:
        original_model: Original HuggingFace model
        mla_model: MLA-converted model
        tokenizer: Tokenizer
        texts: Texts to compare
        max_length: Maximum sequence length
        device: Device for computation

    Returns:
        Dictionary with comparison metrics
    """
    if device is None:
        device = next(mla_model.parameters()).device

    original_model.eval()
    mla_model.eval()

    cosine_similarities = []
    mse_values = []
    kl_divergences = []

    for text in tqdm(texts, desc="Comparing outputs"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            # Get logits from both models
            original_outputs = original_model(**inputs)
            mla_outputs = mla_model(**inputs)

            original_logits = original_outputs.logits
            mla_logits = mla_outputs.logits

            # Flatten logits for comparison
            orig_flat = original_logits.view(-1, original_logits.size(-1))
            mla_flat = mla_logits.view(-1, mla_logits.size(-1))

            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                orig_flat, mla_flat, dim=-1
            ).mean().item()
            cosine_similarities.append(cos_sim)

            # MSE
            mse = torch.nn.functional.mse_loss(mla_flat, orig_flat).item()
            mse_values.append(mse)

            # KL divergence (from original to MLA)
            orig_probs = torch.softmax(original_logits, dim=-1)
            mla_log_probs = torch.log_softmax(mla_logits, dim=-1)
            kl_div = torch.nn.functional.kl_div(
                mla_log_probs.view(-1, mla_logits.size(-1)),
                orig_probs.view(-1, original_logits.size(-1)),
                reduction="batchmean",
            ).item()
            kl_divergences.append(kl_div)

    return {
        "cosine_similarity_mean": sum(cosine_similarities) / len(cosine_similarities),
        "cosine_similarity_min": min(cosine_similarities),
        "mse_mean": sum(mse_values) / len(mse_values),
        "kl_divergence_mean": sum(kl_divergences) / len(kl_divergences),
    }


def generate_samples(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> List[str]:
    """Generate text samples from the model.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompt texts
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        device: Device for computation

    Returns:
        List of generated texts
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    generations = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generations.append(generated_text)

    return generations
