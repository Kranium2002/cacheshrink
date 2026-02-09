"""Benchmark vLLM serving with cacheshrink compressed KV cache.

End-to-end workflow:
1. Convert Mistral 7B (GQA) to MLA with 4x compression
2. Save for both HF and vLLM
3. Compare: HF generate() vs vLLM offline inference (LLM class)
4. Measure throughput, latency, and KV cache memory

Requirements:
    pip install cacheshrink[vllm]
    # Needs GPU with >= 16 GB VRAM

Usage:
    python benchmark_vllm.py
    python benchmark_vllm.py --model mistralai/Mistral-7B-v0.1
    python benchmark_vllm.py --compression-ratio 2
    python benchmark_vllm.py --skip-training  # Use SVD-only initialization
"""

import argparse
import gc
import json
import os
import time

import torch
from datasets import load_dataset

from cacheshrink import (
    MLATrainer,
    compute_perplexity,
    convert_to_mla,
    load_mla_model,
    measure_cache_memory,
    save_mla_model,
)
from cacheshrink.vllm import save_for_vllm


def format_memory(bytes_val):
    """Format bytes to human readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark vLLM + cacheshrink")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name (default: Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=4.0,
        help="KV cache compression ratio (default: 4.0)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip fine-tuning (SVD initialization only)",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts for throughput benchmark (default: 1000)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens to generate per prompt (default: 128)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto-generated from model name)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    MODEL_NAME = args.model
    COMPRESSION_RATIO = args.compression_ratio
    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    model_short = MODEL_NAME.split("/")[-1].lower().replace("-", "_")
    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = f"./{model_short}_mla_{int(COMPRESSION_RATIO)}x"

    HF_PATH = f"{base_dir}_hf"
    VLLM_PATH = f"{base_dir}_vllm"

    print("=" * 70)
    print("cacheshrink vLLM Benchmark")
    print("=" * 70)
    print(f"  Model:             {MODEL_NAME}")
    print(f"  Compression ratio: {COMPRESSION_RATIO}x")
    print(f"  Training:          {'skip' if args.skip_training else f'{args.train_epochs} epochs'}")
    print(f"  HF output:         {HF_PATH}")
    print(f"  vLLM output:       {VLLM_PATH}")

    # =========================================================================
    # Step 1: Convert to MLA (or load existing)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: MLA Conversion")
    print("=" * 70)

    if os.path.exists(HF_PATH):
        print(f"Loading existing HF model from {HF_PATH}...")
        start = time.time()
        mla_model, tokenizer = load_mla_model(HF_PATH, device=DEVICE, dtype=DTYPE)
        print(f"  Loaded in {time.time() - start:.1f}s")
    else:
        print(f"Converting {MODEL_NAME} to MLA...")
        start = time.time()
        mla_model, tokenizer = convert_to_mla(
            MODEL_NAME,
            compression_ratio=COMPRESSION_RATIO,
            device=DEVICE,
            dtype=DTYPE,
            use_calibration=True,
            calibration_dataset="wikitext",
            num_calibration_samples=512,
            use_randomized_svd=False,
            store_original_weights=not args.skip_training,
            verbose=True,
        )
        print(f"  Conversion time: {time.time() - start:.1f}s")

        # Optional training
        if not args.skip_training:
            print("\n  Fine-tuning with reconstruction loss...")
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
            train_texts = [t for t in dataset["train"]["text"] if t and len(t.strip()) > 50][:1000]

            trainer = MLATrainer(
                model=mla_model,
                tokenizer=tokenizer,
                euclidean_lr=1e-5,
                riemannian_lr=1e-4,
                use_distillation=False,
                use_reconstruction_loss=True,
                reconstruction_alpha=0.3,
            )
            train_start = time.time()
            trainer.train(
                train_texts,
                num_epochs=args.train_epochs,
                batch_size=2,
                max_length=1024,
            )
            print(f"  Training time: {time.time() - train_start:.1f}s")
            trainer.cleanup()
            del trainer
            gc.collect()
            torch.cuda.empty_cache()

        # Save HF model
        # print(f"\n  Saving HF model to {HF_PATH}...")
        # save_mla_model(mla_model, tokenizer, HF_PATH)

    print(f"  GPU memory: {get_gpu_memory():.2f} GB")

    mla_config = mla_model.mla_config
    print("\n  MLA Config:")
    print(f"    n_heads:     {mla_config.n_heads}")
    print(f"    n_kv_heads:  {mla_config.n_kv_heads}")
    print(f"    d_head:      {mla_config.d_head}")
    print(f"    d_kv:        {mla_config.d_kv}")
    print(f"    d_latent:    {mla_config.computed_d_latent}")
    print(f"    compression: {mla_config.compression_ratio}x")

    # =========================================================================
    # Step 2: Save for vLLM
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Save for vLLM")
    print("=" * 70)

    if os.path.exists(VLLM_PATH):
        print(f"  vLLM model already exists at {VLLM_PATH}")
    else:
        start = time.time()
        save_for_vllm(mla_model, tokenizer, VLLM_PATH)
        print(f"  Save time: {time.time() - start:.1f}s")

    # Verify config.json
    with open(os.path.join(VLLM_PATH, "config.json")) as f:
        vllm_config = json.load(f)

    d_latent = mla_config.computed_d_latent
    d_head = mla_config.d_head
    n_compressed = d_latent // d_head

    print("\n  vLLM config.json:")
    print(f"    architectures:       {vllm_config.get('architectures')}")
    print(f"    num_key_value_heads: {vllm_config.get('num_key_value_heads')}")
    print(f"    (original was:       {mla_config.n_kv_heads})")
    print(f"    n_compressed_heads:  {n_compressed}")
    print(f"    d_latent={d_latent}, d_head={d_head}")

    # =========================================================================
    # Step 3: KV Cache Compression Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: KV Cache Compression Analysis")
    print("=" * 70)

    cache_stats = measure_cache_memory(mla_model, sequence_lengths=[128, 512, 1024, 2048, 4096])

    print(f"\n  {'Seq Len':<10} {'Standard':<12} {'MLA':<12} {'Saved':<12} {'Ratio':<8}")
    print(f"  {'-' * 54}")
    for seq_len, stats in cache_stats["per_sequence_length"].items():
        print(
            f"  {seq_len:<10} {stats['standard_cache_formatted']:<12} "
            f"{stats['mla_cache_formatted']:<12} "
            f"{stats['memory_saved_formatted']:<12} "
            f"{stats['compression_ratio']:.2f}x"
        )

    # =========================================================================
    # Step 4: HuggingFace Generation Benchmark
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: HuggingFace Generation Benchmark")
    print("=" * 70)

    prompts = [
        "The theory of relativity states that",
        "In machine learning, attention mechanisms",
        "The capital of France is",
        "Quantum computing differs from classical computing because",
        "The most important aspect of software engineering is",
    ]

    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        mla_model.generation_config.pad_token_id = tokenizer.eos_token_id

    # Warmup
    with torch.no_grad():
        inp = tokenizer(prompts[0], return_tensors="pt").to(DEVICE)
        mla_model.generate(inp.input_ids, max_new_tokens=10)

    # Benchmark HF generation
    print(f"\n  Generating {len(prompts)} samples with max_new_tokens={args.max_tokens}...")
    torch.cuda.synchronize()
    hf_start = time.time()
    hf_outputs = []
    with torch.no_grad():
        for prompt in prompts:
            inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            out = mla_model.generate(
                inp.input_ids,
                max_new_tokens=args.max_tokens,
                temperature=0.7,
                do_sample=True,
            )
            hf_outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))
    torch.cuda.synchronize()
    hf_time = time.time() - hf_start
    hf_total_tokens = sum(
        len(tokenizer.encode(o)) - len(tokenizer.encode(p)) for o, p in zip(hf_outputs, prompts)
    )

    print(f"  HF generation time:   {hf_time:.2f}s")
    print(f"  HF tokens generated:  {hf_total_tokens}")
    print(f"  HF throughput:        {hf_total_tokens / hf_time:.1f} tok/s")

    print("\n  Sample outputs:")
    for prompt, output in zip(prompts[:3], hf_outputs[:3]):
        generated = output[len(prompt) :].strip()[:80]
        print(f"    [{prompt[:40]}...] -> {generated}...")

    # =========================================================================
    # Step 5: Perplexity Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Perplexity Evaluation")
    print("=" * 70)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    eval_texts = [t for t in dataset["test"]["text"] if t and len(t.strip()) > 50][:200]

    start = time.time()
    ppl = compute_perplexity(
        mla_model, tokenizer, eval_texts, max_length=512, batch_size=1, verbose=True
    )
    ppl_time = time.time() - start
    print(f"  Perplexity:     {ppl:.2f}")
    print(f"  Eval time:      {ppl_time:.1f}s")

    # =========================================================================
    # Step 6: vLLM Offline Inference Benchmark
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 6: vLLM Offline Inference Benchmark")
    print("=" * 70)

    # Free the HF model first
    del mla_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU memory after cleanup: {get_gpu_memory():.2f} GB")

    try:
        from vllm import LLM, SamplingParams

        print(f"\n  Loading vLLM model from {VLLM_PATH}...")
        start = time.time()
        llm = LLM(
            model=VLLM_PATH,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=2048,
        )
        vllm_load_time = time.time() - start
        print(f"  vLLM load time: {vllm_load_time:.1f}s")

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=args.max_tokens,
        )

        # Generate benchmark prompts
        benchmark_prompts = prompts * (args.num_prompts // len(prompts) + 1)
        benchmark_prompts = benchmark_prompts[: args.num_prompts]

        # Warmup
        print("  Warming up...")
        _ = llm.generate(prompts[:2], sampling_params)

        # Benchmark
        print(f"  Running {len(benchmark_prompts)} prompts, max_tokens={args.max_tokens}...")
        torch.cuda.synchronize()
        vllm_start = time.time()
        vllm_outputs = llm.generate(benchmark_prompts, sampling_params)
        torch.cuda.synchronize()
        vllm_time = time.time() - vllm_start

        vllm_total_tokens = sum(len(o.outputs[0].token_ids) for o in vllm_outputs)
        vllm_throughput = vllm_total_tokens / vllm_time

        print("\n  vLLM Results:")
        print(f"    Prompts:             {len(benchmark_prompts)}")
        print(f"    Total time:          {vllm_time:.2f}s")
        print(f"    Tokens generated:    {vllm_total_tokens}")
        print(f"    Throughput:          {vllm_throughput:.1f} tok/s")
        print(f"    Avg latency/prompt:  {vllm_time / len(benchmark_prompts) * 1000:.1f}ms")

        print("\n  Sample outputs:")
        for i in range(min(3, len(vllm_outputs))):
            prompt = benchmark_prompts[i]
            generated = vllm_outputs[i].outputs[0].text.strip()[:80]
            print(f"    [{prompt[:40]}...] -> {generated}...")

        # Cleanup vLLM
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    except ImportError:
        print("\n  WARNING: vLLM not installed. Skipping vLLM benchmark.")
        print("  Install with: pip install cacheshrink[vllm]")
        vllm_throughput = None
        vllm_time = None

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print(f"\n  Model:              {MODEL_NAME}")
    print(f"  Compression Ratio:  {COMPRESSION_RATIO}x")
    print(f"  Perplexity:         {ppl:.2f}")

    print("\n  KV CACHE:")
    print(f"    Original KV heads:    {mla_config.n_kv_heads}")
    print(f"    Compressed KV heads:  {n_compressed}")
    print(f"    d_latent:             {d_latent}")
    print(f"    Cache compression:    {cache_stats['compression_ratio']:.2f}x")
    if 2048 in cache_stats["per_sequence_length"]:
        s = cache_stats["per_sequence_length"][2048]
        print(
            f"    @2048 tokens:         {s['standard_cache_formatted']} -> {s['mla_cache_formatted']}"
        )

    print("\n  HF GENERATION:")
    print(f"    Throughput:           {hf_total_tokens / hf_time:.1f} tok/s")

    if vllm_throughput is not None:
        print("\n  vLLM SERVING:")
        print(f"    Throughput:           {vllm_throughput:.1f} tok/s")
        print(f"    Speedup vs HF:       {vllm_throughput / (hf_total_tokens / hf_time):.1f}x")

    print("\n  PATHS:")
    print(f"    HF model:  {os.path.abspath(HF_PATH)}")
    print(f"    vLLM model: {os.path.abspath(VLLM_PATH)}")
    print(f"    Serve:     vllm serve {VLLM_PATH}")
    print()


if __name__ == "__main__":
    main()
