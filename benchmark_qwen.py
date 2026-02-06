"""Benchmark Qwen model with MLA conversion, training, and evaluation."""

import time
import torch
import gc
from datasets import load_dataset

from cacheshrink import (
    convert_to_mla,
    save_mla_model,
    load_mla_model,
    MLATrainer,
    compute_perplexity,
    measure_cache_memory,
    generate_samples,
)


def format_memory(bytes_val):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"


def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0


def main():
    print("=" * 70)
    print("Qwen MLA Benchmark (with xKV Cross-Layer Compression)")
    print("=" * 70)

    # Qwen models available (no auth needed):
    # - Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-3B, Qwen/Qwen2.5-7B
    # - Qwen/Qwen2-0.5B, Qwen/Qwen2-1.5B, Qwen/Qwen2-7B
    MODEL_NAME = "Qwen/Qwen2.5-7B"
    COMPRESSION_RATIO = 2.0
    DEVICE = "cuda"
    DTYPE = torch.bfloat16  # bfloat16 needed for Qwen's large bias values

    # Dataset config
    EVAL_SAMPLES = 100
    TRAIN_SAMPLES = 1000
    MAX_LENGTH = 512

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Compression ratio: {COMPRESSION_RATIO}x")
    print(f"  Device: {DEVICE}")
    print(f"  Dtype: {DTYPE}")
    print(f"  Eval samples: {EVAL_SAMPLES}")
    print(f"  Train samples: {TRAIN_SAMPLES}")

    # Load dataset
    print("\n" + "=" * 70)
    print("Loading WikiText-2 dataset...")
    print("=" * 70)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = [t for t in dataset["train"]["text"] if t and len(t.strip()) > 50][:TRAIN_SAMPLES]
    eval_texts = [t for t in dataset["test"]["text"] if t and len(t.strip()) > 50][:EVAL_SAMPLES]

    print(f"  Train texts: {len(train_texts)}")
    print(f"  Eval texts: {len(eval_texts)}")

    # =========================================================================
    # Step 1: Load original model and measure baseline perplexity
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading original Qwen model...")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    start_time = time.time()
    original_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_time = time.time() - start_time
    print(f"  Load time: {load_time:.1f}s")
    print(f"  GPU memory: {get_gpu_memory():.2f} GB")

    # Print attention config
    hf_config = original_model.config
    n_heads = hf_config.num_attention_heads
    n_kv_heads = getattr(hf_config, 'num_key_value_heads', n_heads)
    print(f"\n  Attention Configuration:")
    print(f"    Query heads (n_heads): {n_heads}")
    print(f"    KV heads (n_kv_heads): {n_kv_heads}")
    if n_kv_heads < n_heads:
        print(f"    Type: GQA ({n_heads // n_kv_heads}x repetition)")
    else:
        print(f"    Type: MHA (standard)")

    # Baseline perplexity
    print("\nComputing baseline perplexity...")
    start_time = time.time()
    baseline_ppl = compute_perplexity(
        original_model, tokenizer, eval_texts[:50],
        max_length=MAX_LENGTH, batch_size=1, verbose=True
    )
    ppl_time = time.time() - start_time
    print(f"  Baseline perplexity: {baseline_ppl:.2f}")
    print(f"  Evaluation time: {ppl_time:.1f}s")

    # Generate baseline samples
    print("\nGenerating baseline samples...")
    prompts = [
        "The theory of relativity states that",
        "In machine learning, attention mechanisms",
        "The capital of France is",
    ]
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        original_model.generation_config.pad_token_id = tokenizer.eos_token_id
    baseline_generations = generate_samples(
        original_model, tokenizer, prompts,
        max_new_tokens=50, temperature=0.7
    )
    print("\nBaseline generations:")
    for prompt, gen in zip(prompts, baseline_generations):
        print(f"  Prompt: {prompt}")
        print(f"  Output: {gen[len(prompt):].strip()[:100]}...")
        print()

    # =========================================================================
    # Step 2: Load or Convert to MLA
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Loading/Converting to MLA...")
    print("=" * 70)

    # Clear memory
    print("Freeing original model memory...")
    del original_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory after cleanup: {get_gpu_memory():.2f} GB")

    # Model paths (include xkv in path to distinguish from per-layer MLA)
    import os
    model_short_name = MODEL_NAME.split("/")[-1].lower()
    MLA_CONVERTED_PATH = f"./{model_short_name}-xkv-{int(COMPRESSION_RATIO)}x-converted"
    MLA_TRAINED_PATH = f"./{model_short_name}-xkv-{int(COMPRESSION_RATIO)}x-trained"

    if os.path.exists(MLA_CONVERTED_PATH):
        print(f"Loading existing converted model from {MLA_CONVERTED_PATH}...")
        start_time = time.time()
        mla_model, tokenizer = load_mla_model(MLA_CONVERTED_PATH, device=DEVICE, dtype=DTYPE)
        conversion_time = time.time() - start_time
        print(f"  Load time: {conversion_time:.1f}s")
    else:
        print("No existing model found, converting...")
        start_time = time.time()
        mla_model, tokenizer = convert_to_mla(
            MODEL_NAME,
            compression_ratio=COMPRESSION_RATIO,
            compression_method="auto",  # Auto-detect: uses xKV for GQA models like Qwen
            cross_layer_group_size=4,   # Layers per xKV compression group
            xkv_skip_early_layers=0,    # Skip first 4 layers
            keep_early_layers_original=False,  # Keep early layers as original (no compression)
            device=DEVICE,
            dtype=DTYPE,
            use_calibration=True,
            calibration_dataset="wikitext",
            calibration_dataset_subset="wikitext-2-raw-v1",
            num_calibration_samples=256,
            max_calibration_length=1024,
            use_randomized_svd=False,
            store_original_weights=True,  # Store for reconstruction loss
            verbose=True,
        )
        conversion_time = time.time() - start_time
        print(f"\n  Conversion time: {conversion_time:.1f}s")

        print(f"\nSaving converted model to {MLA_CONVERTED_PATH}...")
        save_mla_model(mla_model, tokenizer, MLA_CONVERTED_PATH)

    print(f"  GPU memory after loading: {get_gpu_memory():.2f} GB")

    # =========================================================================
    # Step 3: Measure KV cache compression
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: KV Cache Compression Analysis")
    print("=" * 70)

    cache_stats = measure_cache_memory(mla_model, sequence_lengths=[128, 512, 1024, 2048, 4096])

    print(f"\nMLA Configuration:")
    print(f"  d_latent: {cache_stats['config']['d_latent']}")
    print(f"  d_kv (original): {cache_stats['config']['d_kv']}")
    print(f"  n_layers: {cache_stats['config']['n_layers']}")
    print(f"  Theoretical compression: {cache_stats['config']['theoretical_compression_ratio']:.2f}x")

    print(f"\nMemory savings by sequence length:")
    print(f"  {'Seq Len':<10} {'Standard':<12} {'MLA':<12} {'Saved':<12} {'Ratio':<8}")
    print(f"  {'-'*54}")
    for seq_len, stats in cache_stats["per_sequence_length"].items():
        print(f"  {seq_len:<10} {stats['standard_cache_formatted']:<12} {stats['mla_cache_formatted']:<12} "
              f"{stats['memory_saved_formatted']:<12} {stats['compression_ratio']:.2f}x")

    # =========================================================================
    # Step 4: Post-conversion perplexity (before training)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Post-conversion perplexity (before training)")
    print("=" * 70)

    start_time = time.time()
    pre_train_ppl = compute_perplexity(
        mla_model, tokenizer, eval_texts[:50],
        max_length=MAX_LENGTH, batch_size=1, verbose=True
    )
    ppl_time = time.time() - start_time
    print(f"  Pre-training perplexity: {pre_train_ppl:.2f}")
    print(f"  Perplexity increase: {((pre_train_ppl / baseline_ppl) - 1) * 100:.1f}%")
    print(f"  Evaluation time: {ppl_time:.1f}s")

    # =========================================================================
    # Step 5: Fine-tune with Riemannian optimization + Reconstruction Loss
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Fine-tuning with Riemannian optimization + Reconstruction Loss")
    print("=" * 70)
    print("Note: Using reconstruction loss (no teacher model needed, saves memory)")

    trainer = MLATrainer(
        model=mla_model,
        tokenizer=tokenizer,
        euclidean_lr=1e-5,
        riemannian_lr=1e-4,
        use_distillation=False,
        use_reconstruction_loss=True,  # Use K/V reconstruction loss
        reconstruction_alpha=0.3,  # Weight of reconstruction loss
    )

    start_time = time.time()
    training_stats = trainer.train(
        train_texts,
        num_epochs=3,
        batch_size=4,
        max_length=MAX_LENGTH,
    )
    train_time = time.time() - start_time
    print(f"\n  Training time: {train_time:.1f}s")
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Step 6: Post-training perplexity
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 6: Post-training perplexity")
    print("=" * 70)

    start_time = time.time()
    post_train_ppl = compute_perplexity(
        mla_model, tokenizer, eval_texts[:50],
        max_length=MAX_LENGTH, batch_size=1, verbose=True
    )
    ppl_time = time.time() - start_time
    print(f"  Post-training perplexity: {post_train_ppl:.2f}")
    print(f"  Improvement from pre-train: {((pre_train_ppl / post_train_ppl) - 1) * 100:.1f}%")
    print(f"  Evaluation time: {ppl_time:.1f}s")

    # =========================================================================
    # Step 7: Generation quality
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 7: Generation quality after MLA conversion + training")
    print("=" * 70)

    mla_generations = generate_samples(
        mla_model, tokenizer, prompts,
        max_new_tokens=50, temperature=0.7
    )

    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        mla_model.generation_config.pad_token_id = tokenizer.eos_token_id

    print("\nMLA model generations:")
    for prompt, gen in zip(prompts, mla_generations):
        print(f"  Prompt: {prompt}")
        print(f"  Output: {gen[len(prompt):].strip()[:100]}...")
        print()

    # =========================================================================
    # Step 8: Orthonormality verification
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 8: Orthonormality verification")
    print("=" * 70)

    max_uk_error = 0
    max_uv_error = 0

    for layer_idx in range(mla_model.mla_config.n_layers):
        attn = mla_model.model.layers[layer_idx].self_attn
        if hasattr(attn, "mla"):
            errors = attn.mla.check_orthonormality()
            max_uk_error = max(max_uk_error, errors["W_uk"][0])
            max_uv_error = max(max_uv_error, errors["W_uv"][0])

    print(f"  Max W_uk orthonormality error: {max_uk_error:.2e}")
    print(f"  Max W_uv orthonormality error: {max_uv_error:.2e}")
    print(f"  Status: {'PASS' if max(max_uk_error, max_uv_error) < 1e-3 else 'WARNING'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Check if xKV was used
    use_xkv = getattr(mla_model.mla_config, 'use_cross_layer', False)
    compression_method = mla_model.mla_config.compression_method
    keep_early_original = getattr(mla_model.mla_config, 'keep_early_layers_original', False)
    skip_layers = getattr(mla_model.mla_config, 'xkv_skip_early_layers', 0)

    # Calculate effective compression
    n_original_layers = skip_layers if keep_early_original else 0
    n_compressed_layers = mla_model.mla_config.n_layers - n_original_layers
    original_total = mla_model.mla_config.n_layers * cache_stats['config']['d_kv'] * 2
    compressed_total = (n_original_layers * cache_stats['config']['d_kv'] * 2 +
                        n_compressed_layers * cache_stats['config']['d_latent'] * 2)
    effective_ratio = original_total / compressed_total

    print(f"""
Model: {MODEL_NAME}
Compression Ratio: {COMPRESSION_RATIO}x (target)
Compression Method: {compression_method}{' (xKV cross-layer)' if use_xkv else ''}
{f'xKV Groups: {mla_model.mla_config.n_groups} groups of {mla_model.mla_config.cross_layer_group_size} layers' if use_xkv else ''}
{f'Early Layers: {skip_layers} layers kept as original (no compression)' if keep_early_original and skip_layers > 0 else ''}

PERPLEXITY:
  Baseline (original):     {baseline_ppl:.2f}
  After MLA conversion:    {pre_train_ppl:.2f} (+{((pre_train_ppl/baseline_ppl)-1)*100:.1f}%)
  After fine-tuning:       {post_train_ppl:.2f} (+{((post_train_ppl/baseline_ppl)-1)*100:.1f}%)

KV CACHE COMPRESSION:
  Original KV dim:         {cache_stats['config']['d_kv'] * 2} (K: {cache_stats['config']['d_kv']}, V: {cache_stats['config']['d_kv']})
  Compressed dim:          {cache_stats['config']['d_latent'] * 2} (c_k: {cache_stats['config']['d_latent']}, c_v: {cache_stats['config']['d_latent']})
  Per-layer compression:   {cache_stats['compression_ratio']:.2f}x
  Effective compression:   {effective_ratio:.2f}x (accounting for {n_original_layers} uncompressed layers)

  Memory at 2048 tokens:   {cache_stats['per_sequence_length'][2048]['standard_cache_formatted']} -> {cache_stats['per_sequence_length'][2048]['mla_cache_formatted']}
  Memory saved:            {cache_stats['per_sequence_length'][2048]['memory_saved_formatted']}

TIMING:
  Conversion time:         {conversion_time:.1f}s
  Training time:           {train_time:.1f}s

ORTHONORMALITY:
  Max error:               {max(max_uk_error, max_uv_error):.2e}
  Status:                  {'PRESERVED' if max(max_uk_error, max_uv_error) < 1e-3 else 'DRIFTED'}
""")

    # =========================================================================
    # Save trained model
    # =========================================================================
    print(f"Saving trained model to {MLA_TRAINED_PATH}...")
    save_mla_model(mla_model, tokenizer, MLA_TRAINED_PATH)
    print(f"Model saved to {MLA_TRAINED_PATH}")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
