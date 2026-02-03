# kvpress

**KV Cache Compression via Multi-Head Latent Attention with Riemannian Optimization**

Achieve **4-16x KV cache compression** on LLaMA, Mistral, GPT-2, and other transformer models while maintaining model quality through mathematically principled compression and fine-tuning.

## Overview

`kvpress` converts HuggingFace transformer models to use **Multi-Head Latent Attention (MLA)**, dramatically reducing KV cache memory during inference. The library uses **Riemannian optimization on Stiefel manifolds** to ensure orthonormality constraints are preserved during fine-tuning, enabling high compression ratios with minimal quality loss.

### Key Features

- **4-16x KV cache compression** - Reduce memory from GBs to MBs for long sequences
- **Drop-in replacement** - Works with existing HuggingFace models and generation pipelines
- **Mathematically principled** - Stiefel manifold constraints ensure stable compression/decompression
- **Calibration-aware initialization** - SVD-based initialization using real activation statistics
- **Knowledge distillation** - Fine-tune compressed models to match original model outputs
- **Multiple model support** - LLaMA 2/3, Mistral, Qwen, GPT-2, and extensible to others

## How It Works

### The KV Cache Problem

In standard transformer attention, the KV cache grows linearly with sequence length:

```
Standard KV Cache Size = 2 × n_layers × seq_len × d_kv × bytes_per_element

For LLaMA-2 7B at 4096 tokens (float16):
= 2 × 32 × 4096 × 4096 × 2 = 2 GB
```

This becomes a major bottleneck for long-context inference and high-throughput serving.

### Multi-Head Latent Attention (MLA)

MLA compresses the KV cache by projecting keys and values into a shared low-dimensional latent space:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Standard Attention                            │
│                                                                  │
│  Hidden States ─┬─► W_k ─► K (d_kv) ──┐                         │
│       (d_model) │                      ├─► Attention ─► Output   │
│                 └─► W_v ─► V (d_kv) ──┘                         │
│                                                                  │
│  Cache stores: K + V = 2 × d_kv per token                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    MLA (Multi-Head Latent Attention)            │
│                                                                  │
│                 ┌─► W_down_k ─► c_k (d_latent) ─► W_uk ─► K     │
│  Hidden States ─┤                                      │        │
│       (d_model) │                                      ├─► Attn │
│                 └─► W_down_v ─► c_v (d_latent) ─► W_uv ─► V     │
│                                                                  │
│  Cache stores: c_k + c_v = 2 × d_latent per token               │
│  Compression: d_kv / d_latent (e.g., 4x, 8x, 16x)               │
└─────────────────────────────────────────────────────────────────┘
```

### Stiefel Manifold Constraints

The decompression matrices `W_uk` and `W_uv` are constrained to have **orthonormal columns** (Stiefel manifold). This ensures:

1. **Stable decompression** - The projection preserves geometric relationships
2. **Energy preservation** - Orthonormal matrices preserve vector norms
3. **Invertibility** - Clean mathematical properties for initialization via SVD

During fine-tuning, we use **Riemannian optimization** (via `geoopt`) to maintain these constraints while updating the parameters.

### Compression Math

```
Original KV dimension:   2 × d_kv     (K and V each have dimension d_kv)
Compressed dimension:    2 × d_latent (c_k and c_v each have dimension d_latent)

Compression ratio = (2 × d_kv) / (2 × d_latent) = d_kv / d_latent

Example (LLaMA-2 7B with 4x compression):
  d_kv = 4096 (32 heads × 128 head_dim)
  d_latent = 1024
  Compression = 4096 / 1024 = 4x
```

## Installation

```bash
pip install kvpress

# For development
git clone https://github.com/your-repo/kvpress
cd kvpress
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- transformers >= 4.35
- geoopt >= 0.5 (for Riemannian optimization)

## Quick Start

### Basic Conversion

```python
from kvpress import convert_to_mla, save_mla_model, load_mla_model

# Convert a HuggingFace model to MLA with 4x compression
model, tokenizer = convert_to_mla(
    "meta-llama/Llama-2-7b-hf",
    compression_ratio=4.0,
    device="cuda",
    dtype=torch.float16,
)

# Generate text (works exactly like the original model)
inputs = tokenizer("The theory of relativity", return_tensors="pt").to("cuda")
outputs = model.generate(inputs.input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))

# Save the compressed model
save_mla_model(model, tokenizer, "./llama-7b-mla-4x")

# Load it later
model, tokenizer = load_mla_model("./llama-7b-mla-4x", device="cuda")
```

### With Calibration (Recommended)

Calibration-aware initialization uses real activation statistics for better compression:

```python
model, tokenizer = convert_to_mla(
    "meta-llama/Llama-2-7b-hf",
    compression_ratio=4.0,
    device="cuda",
    use_calibration=True,                    # Enable calibration
    calibration_dataset="wikitext",          # Dataset for calibration
    calibration_config="wikitext-2-raw-v1",
    num_calibration_samples=128,             # More samples = better init
    max_calibration_length=512,
)
```

### Fine-tuning with Knowledge Distillation

For best results, fine-tune the compressed model using knowledge distillation:

```python
from kvpress import MLATrainer
from transformers import AutoModelForCausalLM

# Load original model as teacher
teacher = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Create trainer with Riemannian optimization
trainer = MLATrainer(
    model=model,
    tokenizer=tokenizer,
    teacher_model=teacher,
    euclidean_lr=1e-5,      # Learning rate for W_down (compression)
    riemannian_lr=1e-4,     # Learning rate for W_uk, W_uv (decompression)
    use_distillation=True,  # Match teacher's output distribution
)

# Train on your data
trainer.train(
    train_texts,           # List of strings or HuggingFace dataset
    num_epochs=10,
    batch_size=4,
    max_length=512,
)
```

### Evaluation

```python
from kvpress import compute_perplexity, measure_cache_memory

# Measure perplexity
ppl = compute_perplexity(model, tokenizer, eval_texts, max_length=512)
print(f"Perplexity: {ppl:.2f}")

# Analyze KV cache compression
stats = measure_cache_memory(model, sequence_lengths=[512, 1024, 2048, 4096])

print(f"Compression ratio: {stats['compression_ratio']:.1f}x")
for seq_len, info in stats['per_sequence_length'].items():
    print(f"  {seq_len} tokens: {info['standard_cache_formatted']} -> {info['mla_cache_formatted']}")
```

## Benchmark Results

### LLaMA-2 7B with 4x Compression

| Metric | Original | After MLA | After Fine-tuning |
|--------|----------|-----------|-------------------|
| Perplexity (WikiText-2) | 11.75 | 16.61 (+41%) | 13.36 (+14%) |
| KV Cache @ 2048 tokens | 1.00 GB | 256 MB | 256 MB |
| Memory Saved | - | 768 MB | 768 MB |

### Memory Savings by Sequence Length (4x compression)

| Sequence Length | Standard Cache | MLA Cache | Saved |
|-----------------|----------------|-----------|-------|
| 512 | 256 MB | 64 MB | 192 MB |
| 1024 | 512 MB | 128 MB | 384 MB |
| 2048 | 1.00 GB | 256 MB | 768 MB |
| 4096 | 2.00 GB | 512 MB | 1.5 GB |
| 8192 | 4.00 GB | 1.00 GB | 3.0 GB |

## Supported Models

| Model Family | Status | Notes |
|--------------|--------|-------|
| LLaMA 2/3 | Supported | Full support including GQA |
| Mistral | Supported | Based on LLaMA handler |
| Qwen/Qwen2 | Supported | Based on LLaMA handler |
| GPT-2 | Supported | Combined QKV projection |

### Adding New Models

Extend `ModelHandler` to support additional architectures:

```python
from kvpress import ModelHandler, register_handler

class MyModelHandler(ModelHandler):
    def get_num_layers(self) -> int:
        return len(self.model.my_layers)

    def extract_qkv_weights(self, layer_idx: int):
        layer = self.model.my_layers[layer_idx]
        return layer.W_q, layer.W_k, layer.W_v, layer.W_o

    # ... implement other methods

register_handler("my_model", MyModelHandler)
```

## API Reference

### Core Functions

```python
# Conversion
convert_to_mla(
    model_name_or_path: str,
    compression_ratio: float = 4.0,      # Target compression (4-16x typical)
    d_latent: int = None,                # Override latent dim (auto if None)
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    use_calibration: bool = True,        # Use activation statistics
    num_calibration_samples: int = 128,
    verbose: bool = True,
) -> Tuple[nn.Module, Tokenizer]

# Save/Load
save_mla_model(model, tokenizer, path: str)
load_mla_model(path: str, device: str = "cuda") -> Tuple[nn.Module, Tokenizer]
```

### Training

```python
MLATrainer(
    model: nn.Module,
    tokenizer: Tokenizer,
    teacher_model: nn.Module = None,     # For distillation
    euclidean_lr: float = 1e-5,          # W_down learning rate
    riemannian_lr: float = 1e-4,         # W_uk, W_uv learning rate
    use_distillation: bool = True,       # Recommended
)
```

### Evaluation

```python
compute_perplexity(model, tokenizer, texts, max_length=512) -> float
measure_cache_memory(model, sequence_lengths=[128, 512, 2048]) -> dict
generate_samples(model, tokenizer, prompts, max_new_tokens=50) -> List[str]
```

### Configuration

```python
MLAConfig(
    model_name: str,
    model_type: str,           # "llama", "mistral", "gpt2", "qwen"
    compression_ratio: float,  # 4.0, 8.0, 16.0, etc.
    d_latent: int = None,      # Auto-computed if None
    n_heads: int,
    n_kv_heads: int,           # For GQA models
    d_model: int,
    d_head: int,
    n_layers: int,
)
```

## Advanced Usage

### Custom Initialization

```python
from kvpress import balanced_svd_init, MLACompression

# Manual SVD initialization
W_down_k, W_down_v, W_uk, W_uv = balanced_svd_init(
    W_k=original_key_weights,
    W_v=original_value_weights,
    d_latent=1024,
)

# Create compression module
compression = MLACompression(config)
compression.init_from_weights(W_down_k, W_down_v, W_uk, W_uv)
```

### Checking Orthonormality

```python
# Verify Stiefel constraints are maintained
for layer_idx in range(model.mla_config.n_layers):
    attn = model.model.layers[layer_idx].self_attn
    errors = attn.mla.check_orthonormality()
    print(f"Layer {layer_idx}: W_uk error={errors['W_uk'][0]:.2e}, W_uv error={errors['W_uv'][0]:.2e}")
```

### Training Configuration

```python
from kvpress import TrainingConfig

config = TrainingConfig(
    euclidean_lr=1e-5,
    riemannian_lr=1e-4,
    num_epochs=10,
    batch_size=4,
    max_length=512,
    use_distillation=True,
    distillation_temperature=2.0,
    distillation_alpha=0.9,           # 90% distillation, 10% LM loss
    check_orthonormality_steps=100,   # Monitor constraint satisfaction
)
```

## How the Training Works

### Two-Optimizer Approach

kvpress uses separate optimizers for different parameter types:

1. **AdamW** for Euclidean parameters (`W_down_k`, `W_down_v`)
   - Standard gradient descent
   - Updates compression matrices freely

2. **RiemannianAdam** for Stiefel parameters (`W_uk`, `W_uv`)
   - Gradients are projected to tangent space of Stiefel manifold
   - Updates follow geodesics to maintain orthonormality
   - No explicit re-orthonormalization needed

### Knowledge Distillation

Instead of training on language modeling loss alone (which can cause the compressed model to drift), we use knowledge distillation:

```
Loss = α × KL(student || teacher) + (1-α) × LM_loss

where:
  - student = softmax(MLA_logits / T)
  - teacher = softmax(original_logits / T)
  - T = temperature (default 2.0)
  - α = distillation weight (default 0.9)
```

This keeps the compressed model's behavior close to the original.

## Citation

If you use kvpress in your research, please cite:

```bibtex
@software{kvpress2024,
  title = {kvpress: KV Cache Compression via Multi-Head Latent Attention},
  year = {2024},
  url = {https://github.com/your-repo/kvpress}
}
```

## License

Apache 2.0
