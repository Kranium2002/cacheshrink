"""Benchmark Mistral 7B (GQA model) with MLA conversion, training, and evaluation.

Mistral 7B uses Grouped Query Attention:
- 32 query heads
- 8 KV heads (4x repetition)
- This tests the GQA support in cacheshrink
"""

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
    print("Mistral 7B MLA Benchmark (GQA Model)")
    print("=" * 70)

    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    COMPRESSION_RATIO = 16.0
    DEVICE = "cuda"
    DTYPE = torch.float16

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
    print("Step 1: Loading original Mistral 7B model...")
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

    # Print GQA info
    hf_config = original_model.config
    print(f"\n  GQA Configuration:")
    print(f"    Query heads (n_heads): {hf_config.num_attention_heads}")
    print(f"    KV heads (n_kv_heads): {hf_config.num_key_value_heads}")
    print(f"    GQA ratio: {hf_config.num_attention_heads // hf_config.num_key_value_heads}x")
    print(f"    Head dim: {hf_config.hidden_size // hf_config.num_attention_heads}")

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

    # Model paths
    import os
    MLA_CONVERTED_PATH = f"./mistral-7b-mla-{int(COMPRESSION_RATIO)}x-converted"
    MLA_TRAINED_PATH = f"./mistral-7b-mla-{int(COMPRESSION_RATIO)}x-trained"

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
            device=DEVICE,
            dtype=DTYPE,
            use_calibration=True,
            calibration_dataset="wikitext",
            calibration_config="wikitext-2-raw-v1",
            num_calibration_samples=256,
            max_calibration_length=1024,
            use_randomized_svd=False,
            verbose=True,
        )
        conversion_time = time.time() - start_time
        print(f"\n  Conversion time: {conversion_time:.1f}s")

        print(f"\nSaving converted model to {MLA_CONVERTED_PATH}...")
        save_mla_model(mla_model, tokenizer, MLA_CONVERTED_PATH)

    print(f"  GPU memory after loading: {get_gpu_memory():.2f} GB")

    # Verify GQA config was preserved
    print(f"\n  MLA Config (GQA preserved):")
    print(f"    n_heads: {mla_model.mla_config.n_heads}")
    print(f"    n_kv_heads: {mla_model.mla_config.n_kv_heads}")
    print(f"    is_gqa: {mla_model.mla_config.is_gqa}")
    print(f"    n_rep: {mla_model.mla_config.n_rep}")
    print(f"    d_kv (actual KV dim): {mla_model.mla_config.d_kv}")
    print(f"    d_latent (compressed): {mla_model.mla_config.d_latent}")

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
    # Step 5: Fine-tune with Riemannian optimization + KL-guided loss
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Fine-tuning with Riemannian optimization + KL-guided loss")
    print("=" * 70)

    # Reload original model as teacher
    print("Loading teacher model for KL-guided training...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    teacher_model.eval()

    trainer = MLATrainer(
        model=mla_model,
        tokenizer=tokenizer,
        teacher_model=teacher_model,
        euclidean_lr=1e-5,
        riemannian_lr=1e-4,
        use_distillation=True,
    )

    start_time = time.time()
    training_stats = trainer.train(
        train_texts,
        num_epochs=40,
        batch_size=4,
        max_length=MAX_LENGTH,
    )
    train_time = time.time() - start_time
    print(f"\n  Training time: {train_time:.1f}s")

    # Clean up teacher model
    del teacher_model
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

    print(f"""
Model: {MODEL_NAME}
Attention Type: GQA ({mla_model.mla_config.n_heads} query heads, {mla_model.mla_config.n_kv_heads} KV heads)
Compression Ratio: {COMPRESSION_RATIO}x

PERPLEXITY:
  Baseline (original):     {baseline_ppl:.2f}
  After MLA conversion:    {pre_train_ppl:.2f} (+{((pre_train_ppl/baseline_ppl)-1)*100:.1f}%)
  After fine-tuning:       {post_train_ppl:.2f} (+{((post_train_ppl/baseline_ppl)-1)*100:.1f}%)

KV CACHE COMPRESSION (GQA-aware):
  Original KV dim:         {cache_stats['config']['d_kv'] * 2} (K: {cache_stats['config']['d_kv']}, V: {cache_stats['config']['d_kv']})
  Compressed dim:          {cache_stats['config']['d_latent'] * 2} (c_k: {cache_stats['config']['d_latent']}, c_v: {cache_stats['config']['d_latent']})
  Actual compression:      {cache_stats['compression_ratio']:.2f}x

  Note: GQA already reduces KV cache vs MHA (8 KV heads vs 32 query heads)
  MLA provides additional {COMPRESSION_RATIO}x on top of GQA savings

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
