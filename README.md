# cacheshrink

**KV Cache Compression via Multi-Head Latent Attention with Riemannian Optimization**

Achieve **2-16x KV cache compression** on LLaMA, Mistral, Qwen, GPT-2, and other transformer models while maintaining model quality through mathematically principled compression and fine-tuning.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Choosing a Compression Method](#choosing-a-compression-method)
- [Usage Guide](#usage-guide)
  - [Converting MHA Models (GPT-2, LLaMA 2)](#converting-mha-models-gpt-2-llama-2)
  - [Converting GQA Models (Qwen, Mistral, LLaMA 3)](#converting-gqa-models-qwen-mistral-llama-3)
  - [Early Layer Skipping (xKV only)](#early-layer-skipping-xkv-only)
  - [Fine-tuning with Reconstruction Loss](#fine-tuning-with-reconstruction-loss)
  - [Fine-tuning with Knowledge Distillation](#fine-tuning-with-knowledge-distillation)
  - [Evaluation](#evaluation)
  - [Saving and Loading](#saving-and-loading)
    - [Reducing Saved Model Size](#reducing-saved-model-size)
- [Parameter Reference](#parameter-reference)
- [Training Details](#training-details)
- [Benchmark Results](#benchmark-results)
- [Supported Models](#supported-models)
- [Adding New Models](#adding-new-models)
- [License](#license)

## Overview

During inference, transformer models store **keys and values** (the KV cache) for every token in the sequence. This cache grows linearly with sequence length and becomes the dominant memory bottleneck for long-context and high-throughput serving:

```
Standard KV Cache Size = 2 x n_layers x seq_len x d_kv x bytes_per_element

Qwen2.5-7B at 4096 tokens (float16):
= 2 x 28 x 4096 x 512 x 2 = 224 MB

LLaMA-2 7B at 4096 tokens (float16):
= 2 x 32 x 4096 x 4096 x 2 = 2 GB
```

`cacheshrink` compresses this cache by converting attention layers to use **Multi-Head Latent Attention (MLA)**. Instead of caching full-dimensional keys and values, it caches compact latent representations and reconstructs K/V on the fly during attention.

### Key Features

- **2-16x KV cache compression** with minimal quality loss
- **Drop-in replacement** for HuggingFace models — generation, evaluation, and pipelines work unchanged
- **Mathematically principled** — Stiefel manifold constraints ensure stable compression/decompression
- **Calibration-aware initialization** — SVD on real activation statistics for high-quality starting point
- **Two training approaches** — reconstruction loss (no teacher model) or knowledge distillation
- **Automatic method selection** — detects MHA/GQA/MQA and picks the optimal compression strategy
- **Cross-layer compression (xKV)** — shared basis vectors across layer groups for GQA models

## Installation

```bash
pip install cacheshrink

# For development
git clone https://github.com/your-repo/cacheshrink
cd cacheshrink
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- transformers >= 4.35
- geoopt >= 0.5 (Riemannian optimization)

## Quick Start

The simplest way to use cacheshrink is with `compression_method="auto"`, which detects your model's attention type and picks the best method:

```python
import torch
from cacheshrink import convert_to_mla, compute_perplexity

# Convert any supported model — auto-selects the best compression method
model, tokenizer = convert_to_mla(
    "Qwen/Qwen2.5-7B",
    compression_ratio=2.0,
    compression_method="auto",
    device="cuda",
    dtype=torch.bfloat16,
    use_calibration=True,
)

# Use it exactly like the original model
inputs = tokenizer("The theory of relativity", return_tensors="pt").to("cuda")
outputs = model.generate(inputs.input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))

# Measure perplexity
ppl = compute_perplexity(model, tokenizer, eval_texts, max_length=512)
```

## How It Works

### Standard Attention vs MLA

In standard attention, hidden states are projected to full-dimensional keys and values, which are cached:

```
Standard Attention:
  h -> W_k -> K (d_kv)  ]
                         ]-> cache stores K + V = 2 x d_kv per token
  h -> W_v -> V (d_kv)  ]
```

MLA adds a compression bottleneck — hidden states are first projected to a small latent space, and only this latent representation is cached. Keys and values are reconstructed on the fly:

```
MLA (Multi-Head Latent Attention):
  h -> W_down_k -> c_k (d_latent) -> W_uk -> K (d_kv)
                         |
                   [cached: c_k]     cache = 2 x d_latent per token
                         |
  h -> W_down_v -> c_v (d_latent) -> W_uv -> V (d_kv)
```

The compression ratio is `d_kv / d_latent`. For example, with d_kv=512 and d_latent=256, you get 2x compression.

### The Matrices

Each compressed attention layer has four learned matrices:

| Matrix | Shape | Role | Optimizer |
|--------|-------|------|-----------|
| `W_down_k` | (d_latent, d_model) | Compresses hidden states to key latent | AdamW |
| `W_down_v` | (d_latent, d_model) | Compresses hidden states to value latent | AdamW |
| `W_uk` | (d_kv, d_latent) | Decompresses key latent back to keys | RiemannianAdam |
| `W_uv` | (d_kv, d_latent) | Decompresses value latent back to values | RiemannianAdam |

The decompression matrices (`W_uk`, `W_uv`) are constrained to have **orthonormal columns** — they live on the Stiefel manifold. This constraint ensures:

- **Stable reconstruction** — orthonormal projections preserve geometric relationships
- **Energy preservation** — vector norms are maintained through decompression
- **Clean initialization** — SVD naturally produces orthonormal matrices

During training, `geoopt`'s RiemannianAdam optimizer updates these matrices while automatically staying on the manifold. No explicit re-orthonormalization is needed.

### Per-Layer vs Cross-Layer Compression

**Per-layer (separate):** Each layer has its own W_down_k, W_down_v, W_uk, W_uv. Works well for MHA models where each layer's KV space is high-dimensional (many KV heads).

**Cross-layer (xKV):** Groups of adjacent layers **share** W_uk and W_uv. Only W_down_k and W_down_v differ per layer. This exploits the fact that adjacent layers in GQA models have similar KV structure, so a shared basis can represent them all efficiently.

```
xKV Group (layers 4-7):
  Shared: W_uk, W_uv           (one pair for the whole group)
  Per-layer: W_down_k, W_down_v (different for each layer)

  Layer 4: h -> W_down_k_4 -> c_k -> [shared W_uk] -> K_4
  Layer 5: h -> W_down_k_5 -> c_k -> [shared W_uk] -> K_5
  Layer 6: h -> W_down_k_6 -> c_k -> [shared W_uk] -> K_6
  Layer 7: h -> W_down_k_7 -> c_k -> [shared W_uk] -> K_7
```

## Choosing a Compression Method

| Method | When to Use | Models |
|--------|-------------|--------|
| `"auto"` | **Always safe.** Auto-detects and picks the best method. | Any |
| `"separate"` | MHA models with many KV heads. Each layer compressed independently. | GPT-2, LLaMA 2 |
| `"xkv"` | GQA/MQA models with few KV heads. Shares basis across layer groups. | Qwen, Mistral, LLaMA 3 |
| `"joint"` | Experimental. Only useful if training from scratch. | N/A |
| `"decoupled_rope"` | Experimental. Preserves positional information separately. | N/A |

**Why do GQA models need xKV?** GQA models like Qwen2.5-7B have only 4 KV heads (d_kv=512). Per-layer SVD on a 512-dimensional space has limited redundancy to exploit. Cross-layer SVD stacks K/V from multiple layers, giving a much larger matrix with more structure to compress.

**Why don't MHA models need xKV?** MHA models like LLaMA-2 7B have 32 KV heads (d_kv=4096). Per-layer SVD already finds ample redundancy within each layer. Forcing layers to share a basis would be unnecessarily constraining.

Use `compression_method="auto"` and it handles this automatically.

## Usage Guide

### Converting MHA Models (GPT-2, LLaMA 2)

For MHA models, use `"separate"` (or `"auto"`, which selects `"separate"` for MHA):

```python
import torch
from cacheshrink import convert_to_mla

model, tokenizer = convert_to_mla(
    "meta-llama/Llama-2-7b-hf",
    compression_ratio=4.0,               # 4x KV cache reduction
    compression_method="auto",           # auto-selects "separate" for MHA
    device="cuda",
    dtype=torch.float16,
    use_calibration=True,                # recommended: uses activation statistics
    num_calibration_samples=128,
    max_calibration_length=512,
    store_original_weights=True,         # needed for reconstruction loss training
    verbose=True,
)
```

### Converting GQA Models (Qwen, Mistral, LLaMA 3)

For GQA models, use `"auto"` or `"xkv"`:

```python
import torch
from cacheshrink import convert_to_mla

model, tokenizer = convert_to_mla(
    "Qwen/Qwen2.5-7B",
    compression_ratio=2.0,               # 2x KV cache reduction
    compression_method="auto",           # auto-selects "xkv" for GQA
    cross_layer_group_size=4,            # 4 layers share decompression matrices
    device="cuda",
    dtype=torch.bfloat16,
    use_calibration=True,
    num_calibration_samples=256,
    max_calibration_length=1024,
    store_original_weights=True,
    verbose=True,
)
```

### Early Layer Skipping (xKV only)

Early transformer layers (layers 0-3 typically) capture low-level features where information is spread uniformly across dimensions, making compression lossier. You can skip these layers to preserve quality at a small cost to compression ratio:

```python
model, tokenizer = convert_to_mla(
    "Qwen/Qwen2.5-7B",
    compression_ratio=2.0,
    compression_method="auto",
    cross_layer_group_size=4,
    xkv_skip_early_layers=4,             # don't compress layers 0-3
    keep_early_layers_original=True,     # keep them as original attention
    device="cuda",
    dtype=torch.bfloat16,
    use_calibration=True,
    store_original_weights=True,
)
```

**What the parameters mean:**

- `xkv_skip_early_layers=4` — the first 4 layers are excluded from xKV compression groups
- `keep_early_layers_original=True` — those skipped layers keep their original attention (no compression at all). If `False`, they would get per-layer MLA compression instead.

**Trade-off:** Skipping 4 out of 28 layers reduces effective compression from 2.0x to ~1.75x, but significantly improves quality (see [Benchmark Results](#benchmark-results)).

**These parameters are xKV-only.** Passing them with non-xKV methods (`"separate"`, `"joint"`, `"decoupled_rope"`) raises a `ValueError`.

### Important: Use bfloat16, Not float16

Always use `dtype=torch.bfloat16` (not `torch.float16`) when converting and evaluating models. Qwen and other large models have weight values that exceed float16's narrow dynamic range (max ~65,504), causing overflow to `inf`/`NaN` during inference. bfloat16 has the same exponent range as float32 and handles these values correctly.

```python
# GOOD
model, tokenizer = convert_to_mla("Qwen/Qwen2.5-7B", dtype=torch.bfloat16, ...)

# BAD — will produce NaN logits and inf perplexity
model, tokenizer = convert_to_mla("Qwen/Qwen2.5-7B", dtype=torch.float16, ...)
```

This also applies when loading saved models:

```python
model, tokenizer = load_mla_model("./my-model", device="cuda", dtype=torch.bfloat16)
```

### Fine-tuning with Reconstruction Loss

Reconstruction loss directly optimizes the compression matrices to minimize the difference between original and reconstructed K/V projections. This is the recommended training approach because it doesn't require loading a teacher model:

```python
from cacheshrink import MLATrainer

trainer = MLATrainer(
    model=model,
    tokenizer=tokenizer,
    euclidean_lr=1e-5,              # learning rate for W_down_k, W_down_v
    riemannian_lr=1e-4,             # learning rate for W_uk, W_uv (on Stiefel manifold)
    use_distillation=False,
    use_reconstruction_loss=True,   # optimize K/V reconstruction directly
    reconstruction_alpha=0.3,       # blend: 70% LM loss + 30% reconstruction loss
)

stats = trainer.train(
    train_texts,                    # list of strings or HuggingFace dataset
    num_epochs=3,
    batch_size=4,
    max_length=512,
)
```

**Requirements:** The model must be converted with `store_original_weights=True`. This stores the original W_k and W_v weight matrices as frozen buffers so the reconstruction loss can compute what K/V *should* be.

**How it works:**

```
Total loss = (1 - alpha) x LM_loss + alpha x reconstruction_loss

reconstruction_loss = avg over layers of:
    MSE(K_reconstructed, K_original) + MSE(V_reconstructed, V_original)

where:
    K_original = h @ W_k_original.T + b_k     (what the original model produces)
    K_reconstructed = W_uk @ W_down_k(h) + b_k (what the compressed model produces)
```

### Fine-tuning with Knowledge Distillation

Alternatively, train the compressed model to match the original model's output distribution. This requires 2x GPU memory (both models loaded simultaneously):

```python
from cacheshrink import MLATrainer

trainer = MLATrainer(
    model=model,
    tokenizer=tokenizer,
    euclidean_lr=1e-5,
    riemannian_lr=1e-4,
    use_distillation=True,          # match teacher's output distribution
    # teacher_model is auto-loaded from model.mla_config.model_name
    # or pass teacher_model=your_model explicitly
)

stats = trainer.train(train_texts, num_epochs=3, batch_size=4, max_length=512)
```

**How it works:**

```
Total loss = alpha x KL(student || teacher) + (1 - alpha) x LM_loss

where:
    student = softmax(compressed_model_logits / T)
    teacher = softmax(original_model_logits / T)
    T = temperature (default 2.0)
    alpha = distillation weight (default 0.9)
```

**When to use distillation vs reconstruction loss:**

| | Reconstruction Loss | Knowledge Distillation |
|--|--|--|
| GPU memory | 1x model | 2x model (student + teacher) |
| Setup | Simpler (just `store_original_weights=True`) | Need teacher model loaded |
| What it optimizes | K/V reconstruction fidelity | Output distribution match |
| Best for | Most cases | When output distribution matters more than K/V accuracy |

### Evaluation

```python
from cacheshrink import compute_perplexity, measure_cache_memory, generate_samples

# Perplexity on evaluation data
ppl = compute_perplexity(model, tokenizer, eval_texts, max_length=512, batch_size=1)
print(f"Perplexity: {ppl:.2f}")

# KV cache memory analysis
stats = measure_cache_memory(model, sequence_lengths=[512, 1024, 2048, 4096])
for seq_len, info in stats['per_sequence_length'].items():
    print(f"  {seq_len} tokens: {info['standard_cache_formatted']} -> {info['mla_cache_formatted']}")

# Text generation
outputs = generate_samples(
    model, tokenizer,
    prompts=["The theory of relativity states that"],
    max_new_tokens=100,
    temperature=0.7,
)
```

### Saving and Loading

```python
from cacheshrink import save_mla_model, load_mla_model

# Save (uses safetensors by default)
save_mla_model(model, tokenizer, "./my-compressed-model")

# Load
model, tokenizer = load_mla_model("./my-compressed-model", device="cuda", dtype=torch.bfloat16)
```

Saved files include the model weights, tokenizer, MLA config, and xKV group state (if applicable). Shared tensor references (xKV's shared W_uk/W_uv) are correctly preserved through save/load.

#### Reducing Saved Model Size

By default, `save_mla_model` excludes training buffers (W_k_original, W_v_original) from the saved checkpoint. These buffers store the original K/V projection weights used only for reconstruction loss during training. Excluding them significantly reduces saved model size:

| Model | With Training Buffers | Without (default) | Saved |
|-------|----------------------|-------------------|-------|
| LLaMA-2 7B (16x) | ~14.5 GB | ~12.5 GB | ~2 GB |
| Qwen2.5-7B (2x) | ~15.2 GB | ~15.0 GB | ~200 MB |

If you need to resume training with reconstruction loss later, save with buffers explicitly:

```python
# For distribution/inference (default) — smallest size
save_mla_model(model, tokenizer, "./my-model")

# To resume reconstruction-loss training later — includes original K/V weights
save_mla_model(model, tokenizer, "./my-model-resumable", save_training_buffers=True)
```

At load time, training buffers are also skipped by default — both via `load_mla_model()` and `AutoModelForCausalLM.from_pretrained()`:

```python
# Inference (default) — skips training buffers, lower GPU memory
model, tokenizer = load_mla_model("./my-model", device="cuda", dtype=torch.bfloat16)

# Resume training — loads training buffers
model, tokenizer = load_mla_model("./my-model-resumable", device="cuda", dtype=torch.bfloat16,
                                   load_training_buffers=True)
```

## Parameter Reference

### `convert_to_mla()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | str | *required* | HuggingFace model name or local path |
| `compression_ratio` | float | 4.0 | Target compression ratio. d_latent = d_kv / ratio |
| `compression_method` | str | `"separate"` | `"auto"`, `"separate"`, `"xkv"`, `"joint"`, `"decoupled_rope"` |
| `d_latent` | int | None | Override latent dimension (auto-computed from ratio if None) |
| `device` | str | `"cuda"` | Device to load model on |
| `dtype` | torch.dtype | None | Model dtype (e.g., `torch.bfloat16`) |
| `use_calibration` | bool | True | Use activation statistics for SVD initialization |
| `calibration_dataset` | str | `"wikitext"` | HuggingFace dataset for calibration |
| `calibration_dataset_subset` | str | `"wikitext-2-raw-v1"` | Dataset subset name (second arg to `load_dataset()`) |
| `num_calibration_samples` | int | 128 | Number of calibration samples (more = better init) |
| `max_calibration_length` | int | 512 | Max sequence length for calibration |
| `store_original_weights` | bool | False | Store W_k/W_v for reconstruction loss training (required for `use_reconstruction_loss`) |
| `cross_layer_group_size` | int | 4 | Layers per xKV group (xKV only) |
| `xkv_skip_early_layers` | int | 0 | Number of early layers to exclude from xKV (xKV only) |
| `keep_early_layers_original` | bool | False | Keep skipped layers as original attention (xKV only) |
| `verbose` | bool | True | Print progress and stats |

### `MLATrainer()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | *required* | MLA-converted model |
| `tokenizer` | Tokenizer | *required* | HuggingFace tokenizer |
| `euclidean_lr` | float | 1e-5 | Learning rate for W_down_k, W_down_v |
| `riemannian_lr` | float | 1e-4 | Learning rate for W_uk, W_uv (Stiefel manifold) |
| `use_distillation` | bool | True | Use knowledge distillation (needs teacher model) |
| `use_reconstruction_loss` | bool | False | Use K/V reconstruction loss |
| `reconstruction_alpha` | float | 0.3 | Weight of reconstruction loss (0.0-1.0) |
| `teacher_model` | nn.Module | None | Teacher model for distillation (auto-loaded if None) |

`use_distillation` and `use_reconstruction_loss` are mutually exclusive.

### `trainer.train()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | list or Dataset | *required* | List of strings or HuggingFace dataset |
| `num_epochs` | int | 3 | Number of training epochs |
| `batch_size` | int | 4 | Training batch size |
| `max_length` | int | 512 | Maximum sequence length |

### `save_mla_model()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | *required* | MLA-converted model |
| `tokenizer` | Tokenizer | *required* | HuggingFace tokenizer |
| `save_directory` | str | *required* | Directory to save to |
| `training_stats` | dict | None | Optional training statistics to save |
| `use_safetensors` | bool | True | Use safetensors format (recommended) |
| `enable_hf_loading` | bool | True | Enable `AutoModelForCausalLM.from_pretrained()` support |
| `save_training_buffers` | bool | False | Save W_k_original/W_v_original. Set True only to resume reconstruction-loss training. |

### `load_mla_model()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `load_directory` | str | *required* | Directory containing saved model |
| `device` | str | None | Device to load model to (e.g., `"cuda"`) |
| `dtype` | torch.dtype | None | Model dtype (e.g., `torch.bfloat16`) |
| `low_cpu_mem_usage` | bool | True | Use accelerate-style loading for lower peak memory |
| `load_training_buffers` | bool | False | Load W_k_original/W_v_original. Set True only to resume reconstruction-loss training. |

## Training Details

### What Gets Trained

Only the compression matrices are trained. All other model parameters (embeddings, layer norms, Q/O projections, FFN layers) are frozen:

```
Trainable:
  W_down_k, W_down_v  (per-layer, Euclidean params)  -> AdamW optimizer
  W_uk, W_uv          (shared in xKV, Stiefel params) -> RiemannianAdam optimizer

Frozen:
  Everything else (embeddings, attention Q/O, FFN, layer norms)
```

Typical trainable parameter counts:

| Model | Method | Trainable | Total | % |
|-------|--------|-----------|-------|---|
| Qwen2.5-7B | xKV 2x | ~708M | 7.6B | 9.3% |
| Qwen2.5-7B | xKV 2x (skip 4) | ~826M | 7.6B | 10.9% |
| LLaMA-2 7B | separate 4x | ~537M | 6.7B | 8.0% |

### Riemannian Optimization

The decompression matrices W_uk and W_uv must maintain orthonormal columns (Stiefel manifold constraint: W.T @ W = I). Standard gradient descent would violate this constraint.

`geoopt`'s RiemannianAdam handles this by:
1. Computing the Euclidean gradient
2. Projecting it onto the tangent space of the Stiefel manifold
3. Taking a step along a geodesic (retraction)

This keeps the matrices on the manifold at every step. After training, orthonormality errors are typically < 1e-6.

### Reconstruction Loss vs LM Loss

The training loss is a weighted blend:

- **LM loss** (language modeling cross-entropy) — keeps the model's text generation quality
- **Reconstruction loss** (MSE on K/V) — directly improves compression fidelity

With `reconstruction_alpha=0.3`, the total loss is:
```
loss = 0.7 * LM_loss + 0.3 * reconstruction_loss
```

Higher alpha puts more weight on K/V reconstruction accuracy. Lower alpha preserves more of the original model's generation behavior. 0.3 is a good default.

## Benchmark Results

### Qwen2.5-7B with xKV 2x Compression

| Configuration | Baseline | After Conversion | After 3 Epochs | Degradation |
|--------------|----------|-----------------|----------------|-------------|
| xKV (skip 4 early layers) | 10.58 | 17.44 | **12.46** | +17.8% |
| xKV (all layers compressed) | 10.58 | 46.67 | **14.29** | +35.1% |

Training: 1000 WikiText-2 samples, 3 epochs, batch_size=4, reconstruction_alpha=0.3

**Key observations:**
- Skipping 4 early layers improves post-training perplexity by 1.8 points (12.46 vs 14.29)
- The quality gap with skipping is much smaller (17.8% vs 35.1% degradation from baseline)
- Effective compression is 1.75x with skip vs 2.0x without — a good trade-off

### Mistral 7B with 4x Compression

| Metric | Original | After MLA | After Fine-tuning |
|--------|----------|-----------|-------------------|
| Perplexity (WikiText-2) | 12.90 | 21.84 (+69%) | 13.00 (+0.8%) |
| KV Cache @ 2048 tokens | 512 MB | 128 MB | 128 MB |

Training: 1000 WikiText-2 samples, 3 epochs, batch_size=4, reconstruction_alpha=0.3. **98.9% of conversion degradation recovered.**

### LLaMA-2 7B with 16x Compression

| Metric | Original | After MLA | After Fine-tuning |
|--------|----------|-----------|-------------------|
| Perplexity (WikiText-2) | 11.70 | 255.90 (+2088%) | 17.68 (+51.2%) |
| KV Cache @ 2048 tokens | 1.00 GB | 64 MB | 64 MB |

Training: 1000 WikiText-2 samples, 3 epochs, batch_size=4, reconstruction_alpha=0.3. Even at extreme 16x compression (4096 → 256 dims), fine-tuning recovers most quality — perplexity drops from 255.90 to 17.68.

### Memory Savings

**Qwen2.5-7B (2x compression, 28 layers):**

| Sequence Length | Standard Cache | MLA Cache | Saved |
|-----------------|----------------|-----------|-------|
| 512 | 28 MB | 14 MB | 14 MB |
| 2048 | 112 MB | 56 MB | 56 MB |
| 4096 | 224 MB | 112 MB | 112 MB |

**LLaMA-2 7B (16x compression, 32 layers):**

| Sequence Length | Standard Cache | MLA Cache | Saved |
|-----------------|----------------|-----------|-------|
| 512 | 256 MB | 16 MB | 240 MB |
| 2048 | 1.00 GB | 64 MB | 960 MB |
| 4096 | 2.00 GB | 128 MB | 1.9 GB |
| 8192 | 4.00 GB | 256 MB | 3.75 GB |

## Supported Models

### Tested Open-Source Models

| Model | HuggingFace ID | Attention | KV Heads | Recommended Method |
|-------|----------------|-----------|----------|-------------------|
| GPT-2 | `openai-community/gpt2` | MHA | 12 | `separate` |
| GPT-2 Medium | `openai-community/gpt2-medium` | MHA | 16 | `separate` |
| GPT-2 Large | `openai-community/gpt2-large` | MHA | 20 | `separate` |
| GPT-2 XL | `openai-community/gpt2-xl` | MHA | 25 | `separate` |
| LLaMA 2 7B | `meta-llama/Llama-2-7b-hf` | MHA | 32 | `separate` |
| LLaMA 2 13B | `meta-llama/Llama-2-13b-hf` | MHA | 40 | `separate` |
| LLaMA 2 70B | `meta-llama/Llama-2-70b-hf` | GQA | 8 | `xkv` |
| LLaMA 3 8B | `meta-llama/Meta-Llama-3-8B` | GQA | 8 | `xkv` |
| LLaMA 3 70B | `meta-llama/Meta-Llama-3-70B` | GQA | 8 | `xkv` |
| LLaMA 3.1 8B | `meta-llama/Llama-3.1-8B` | GQA | 8 | `xkv` |
| LLaMA 3.1 70B | `meta-llama/Llama-3.1-70B` | GQA | 8 | `xkv` |
| Mistral 7B v0.1 | `mistralai/Mistral-7B-v0.1` | GQA | 8 | `xkv` |
| Mistral 7B v0.3 | `mistralai/Mistral-7B-v0.3` | GQA | 8 | `xkv` |
| Mixtral 8x7B | `mistralai/Mixtral-8x7B-v0.1` | GQA | 8 | `xkv` |
| Qwen2 7B | `Qwen/Qwen2-7B` | GQA | 4 | `xkv` |
| Qwen2.5 7B | `Qwen/Qwen2.5-7B` | GQA | 4 | `xkv` |
| Qwen2.5 14B | `Qwen/Qwen2.5-14B` | GQA | 4 | `xkv` |
| Qwen2.5 32B | `Qwen/Qwen2.5-32B` | GQA | 8 | `xkv` |
| Qwen2.5 72B | `Qwen/Qwen2.5-72B` | GQA | 8 | `xkv` |

Instruction-tuned variants (e.g., `Llama-2-7b-chat-hf`, `Mistral-7B-Instruct-v0.3`, `Qwen2.5-7B-Instruct`) are also supported — they share the same architecture as their base models.

### Model Family Summary

| Family | Handler | Attention Types | Notes |
|--------|---------|----------------|-------|
| GPT-2 | `gpt2` (dedicated) | MHA | Combined QKV projection, learned position embeddings |
| LLaMA | `generic` | MHA (2-7B/13B), GQA (2-70B, 3.x) | Separate Q/K/V/O projections, RoPE |
| Mistral | `generic` | GQA | Same as LLaMA with sliding window attention |
| Qwen/Qwen2 | `generic` | GQA | Has K/V biases (handled automatically) |
| StableLM | `generic` | GQA | Auto-discovered via GenericHandler |
| Other | `generic` | Any | GenericHandler auto-discovers layer structure via known paths + BFS fallback |
| DeepSeek V2 | N/A | Native MLA | Already uses MLA, not supported |

GPT-2 is the only model with a dedicated handler due to its unique Conv1D-based combined QKV projection and distinct forward signature. All other architectures are handled by `GenericHandler`, which auto-discovers layer lists, attention modules, projection styles, and bias handling at init time.

### Not Supported

Models that already use compressed KV caching (e.g., DeepSeek V2/V3 with native MLA) are detected and rejected with a clear error message. Unrecognized architectures fall through to `GenericHandler`, which auto-discovers the model's layer structure. If auto-discovery fails, see [Adding New Models](#adding-new-models) to add support.

**Attention type glossary:**

- **MHA (Multi-Head Attention):** n_kv_heads == n_heads. Every query head has its own KV head. Large d_kv.
- **GQA (Grouped Query Attention):** 1 < n_kv_heads < n_heads. Multiple query heads share each KV head. Smaller d_kv.
- **MQA (Multi-Query Attention):** n_kv_heads == 1. All query heads share a single KV head. Smallest d_kv.

## Adding New Models

Extend `ModelHandler` to support additional architectures:

```python
from cacheshrink.model_handlers.base import ModelHandler

class MyModelHandler(ModelHandler):
    def get_num_layers(self) -> int:
        return len(self.model.my_layers)

    def get_attention_module(self, layer_idx: int):
        return self.model.my_layers[layer_idx].attention

    def extract_qkv_weights(self, layer_idx: int):
        attn = self.get_attention_module(layer_idx)
        return attn.W_q.data, attn.W_k.data, attn.W_v.data, attn.W_o.data

    def extract_qkv_biases(self, layer_idx: int):
        attn = self.get_attention_module(layer_idx)
        return attn.b_q, attn.b_k, attn.b_v, attn.b_o  # None if no biases

    def replace_attention(self, layer_idx: int, new_attn):
        self.model.my_layers[layer_idx].attention = new_attn
```

Register it in `model_handlers/__init__.py` to make it available via `get_handler()`.

## Roadmap

- **HuggingFace Hub integration** — Save and load compressed models with `AutoModelForCausalLM.from_pretrained()` via `trust_remote_code=True`. Publish compressed models to the Hub for anyone to use without installing cacheshrink.
- **vLLM support** — Custom model plugin for vLLM so compressed models can serve with PagedAttention, continuous batching, and all the production features vLLM provides.
- **Custom Triton kernels** — Fused compress/decompress + attention kernels to eliminate the overhead of reconstructing K/V into a separate buffer. Directly compute attention from the latent representations.
- **Combining with KV cache quantization** — Stack dimensional compression (cacheshrink) with precision reduction (FP8/INT8) for even higher compression ratios.

## License

Apache 2.0
