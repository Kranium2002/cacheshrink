"""Benchmark: GenericHandler with stabilityai/stablelm-2-1_6b (1.6B).

Demonstrates cacheshrink working on a model with NO dedicated handler,
using the auto-discovered GenericHandler fallback.

Model: stabilityai/stablelm-2-1_6b — 1.6B params, model_type="stablelm"
  - GQA: 32 query heads, 4 KV heads
  - separate q_proj / k_proj / v_proj / o_proj
  - RoPE (partial_rotary_factor=0.25)
"""

import time
import gc
import torch
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


MODEL_NAME = "stabilityai/stablelm-2-1_6b"
COMPRESSION_RATIO = 8.0
DEVICE = "cuda"
DTYPE = torch.bfloat16

EVAL_SAMPLES = 50
TRAIN_SAMPLES = 500
MAX_LENGTH = 512


def gpu_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def fmt_mem(b):
    for u in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.2f} {u}"
        b /= 1024
    return f"{b:.2f} TB"


def separator(title):
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def main():
    separator("StableLM-2-1.6B  GenericHandler Benchmark")
    print(f"  Model          : {MODEL_NAME}")
    print(f"  Compression    : {COMPRESSION_RATIO}x")
    print(f"  Device / dtype : {DEVICE} / {DTYPE}")
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    separator("Loading WikiText-2 dataset")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = [t for t in dataset["train"]["text"]
                   if t and len(t.strip()) > 50][:TRAIN_SAMPLES]
    eval_texts = [t for t in dataset["test"]["text"]
                  if t and len(t.strip()) > 50][:EVAL_SAMPLES]
    print(f"  Train texts : {len(train_texts)}")
    print(f"  Eval texts  : {len(eval_texts)}")

    # ------------------------------------------------------------------
    # Step 1 – Load original model, measure baseline
    # ------------------------------------------------------------------
    separator("Step 1: Load original model & measure baseline")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    original_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Load time : {time.time()-t0:.1f}s")
    print(f"  GPU mem   : {gpu_mem():.2f} GB")

    hf_cfg = original_model.config
    n_heads = hf_cfg.num_attention_heads
    n_kv    = getattr(hf_cfg, "num_key_value_heads", n_heads)
    print(f"  Heads     : {n_heads} query, {n_kv} KV  "
          f"({'GQA ' + str(n_heads//n_kv) + 'x' if n_kv < n_heads else 'MHA'})")
    print(f"  model_type: {hf_cfg.model_type}  (no dedicated handler → GenericHandler)")

    # Baseline perplexity
    print("\n  Computing baseline perplexity …")
    t0 = time.time()
    baseline_ppl = compute_perplexity(
        original_model, tokenizer, eval_texts,
        max_length=MAX_LENGTH, batch_size=1, verbose=True,
    )
    print(f"  Baseline PPL : {baseline_ppl:.2f}  ({time.time()-t0:.1f}s)")

    # Baseline generation
    prompts = [
        "The theory of relativity states that",
        "In machine learning, attention mechanisms",
        "The capital of France is",
    ]
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        original_model.generation_config.pad_token_id = tokenizer.eos_token_id
    baseline_gens = generate_samples(
        original_model, tokenizer, prompts,
        max_new_tokens=60, temperature=0.7,
    )
    print("\n  Baseline generations:")
    for p, g in zip(prompts, baseline_gens):
        print(f"    > {p}")
        print(f"      {g[len(p):].strip()[:120]}")

    # ------------------------------------------------------------------
    # Step 2 – Convert to MLA via GenericHandler
    # ------------------------------------------------------------------
    separator("Step 2: Convert to MLA (GenericHandler)")

    del original_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU mem after cleanup: {gpu_mem():.2f} GB")

    import os
    short = MODEL_NAME.split("/")[-1].lower()
    CONVERTED = f"./{short}-mla-{int(COMPRESSION_RATIO)}x-converted"
    TRAINED   = f"./{short}-mla-{int(COMPRESSION_RATIO)}x-trained"

    already_trained = False

    if os.path.exists(TRAINED):
        print(f"  Loading trained model from {TRAINED} …")
        t0 = time.time()
        mla_model, tokenizer = load_mla_model(TRAINED, device=DEVICE, dtype=DTYPE)
        already_trained = True
        print(f"  Load time : {time.time()-t0:.1f}s")
    elif os.path.exists(CONVERTED):
        print(f"  Loading converted model from {CONVERTED} …")
        t0 = time.time()
        mla_model, tokenizer = load_mla_model(CONVERTED, device=DEVICE, dtype=DTYPE)
        print(f"  Load time : {time.time()-t0:.1f}s")
    else:
        print("  Converting from scratch …")
        t0 = time.time()
        mla_model, tokenizer = convert_to_mla(
            MODEL_NAME,
            compression_ratio=COMPRESSION_RATIO,
            compression_method="auto",
            cross_layer_group_size=4,
            xkv_skip_early_layers=0,
            keep_early_layers_original=False,
            device=DEVICE,
            dtype=DTYPE,
            use_calibration=True,
            calibration_dataset="wikitext",
            calibration_dataset_subset="wikitext-2-raw-v1",
            num_calibration_samples=128,
            max_calibration_length=512,
            use_randomized_svd=True,
            store_original_weights=True,
            verbose=True,
        )
        conversion_time = time.time() - t0
        print(f"\n  Conversion time : {conversion_time:.1f}s")
        print(f"  GPU mem         : {gpu_mem():.2f} GB")

        # Verify GenericHandler was used
        handler_type = type(mla_model.mla_handler).__name__
        print(f"  Handler used    : {handler_type}")
        assert handler_type == "GenericHandler", f"Expected GenericHandler, got {handler_type}"

        print(f"\n  Saving to {CONVERTED} …")
        save_mla_model(mla_model, tokenizer, CONVERTED)

    print(f"  GPU mem: {gpu_mem():.2f} GB")

    # ------------------------------------------------------------------
    # Step 3 – KV cache compression stats
    # ------------------------------------------------------------------
    separator("Step 3: KV cache compression stats")

    cache_stats = measure_cache_memory(
        mla_model, sequence_lengths=[128, 512, 1024, 2048],
    )
    cfg = cache_stats["config"]
    print(f"  d_latent (compressed)  : {cfg['d_latent']}")
    print(f"  d_kv     (original)    : {cfg['d_kv']}")
    print(f"  Theoretical ratio      : {cfg['theoretical_compression_ratio']:.2f}x")
    print()
    print(f"  {'Seq Len':<10} {'Original':<12} {'MLA':<12} {'Saved':<12} {'Ratio':<8}")
    print(f"  {'-'*54}")
    for sl, st in cache_stats["per_sequence_length"].items():
        print(f"  {sl:<10} {st['standard_cache_formatted']:<12} "
              f"{st['mla_cache_formatted']:<12} "
              f"{st['memory_saved_formatted']:<12} "
              f"{st['compression_ratio']:.2f}x")

    pre_train_ppl = None
    train_time = 0

    if not already_trained:
        # --------------------------------------------------------------
        # Step 4 – Pre-training perplexity
        # --------------------------------------------------------------
        separator("Step 4: Pre-training perplexity")
        t0 = time.time()
        pre_train_ppl = compute_perplexity(
            mla_model, tokenizer, eval_texts,
            max_length=MAX_LENGTH, batch_size=1, verbose=True,
        )
        print(f"  Pre-train PPL     : {pre_train_ppl:.2f}")
        print(f"  Increase vs base  : +{((pre_train_ppl/baseline_ppl)-1)*100:.1f}%")
        print(f"  Eval time         : {time.time()-t0:.1f}s")

        # --------------------------------------------------------------
        # Step 5 – Fine-tune with reconstruction loss
        # --------------------------------------------------------------
        separator("Step 5: Fine-tune (Riemannian opt + reconstruction loss)")
        print("  No teacher model needed — uses stored original K/V weights")

        trainer = MLATrainer(
            model=mla_model,
            tokenizer=tokenizer,
            euclidean_lr=1e-5,
            riemannian_lr=1e-4,
            use_distillation=False,
            use_reconstruction_loss=True,
            reconstruction_alpha=0.3,
        )

        t0 = time.time()
        trainer.train(
            train_texts,
            num_epochs=3,
            batch_size=2,
            max_length=MAX_LENGTH,
        )
        train_time = time.time() - t0
        print(f"\n  Training time: {train_time:.1f}s")

        gc.collect()
        torch.cuda.empty_cache()
    else:
        separator("Steps 4-5: Skipped (loaded pre-trained)")

    # ------------------------------------------------------------------
    # Step 6 – Post-training perplexity
    # ------------------------------------------------------------------
    separator("Step 6: Post-training perplexity")
    t0 = time.time()
    post_ppl = compute_perplexity(
        mla_model, tokenizer, eval_texts,
        max_length=MAX_LENGTH, batch_size=1, verbose=True,
    )
    print(f"  Post-train PPL   : {post_ppl:.2f}")
    if pre_train_ppl is not None:
        print(f"  Improvement      : {((pre_train_ppl/post_ppl)-1)*100:.1f}% vs pre-train")
    print(f"  vs baseline      : +{((post_ppl/baseline_ppl)-1)*100:.1f}%")
    print(f"  Eval time        : {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 7 – Generation after MLA
    # ------------------------------------------------------------------
    separator("Step 7: MLA generation quality")

    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        mla_model.generation_config.pad_token_id = tokenizer.eos_token_id

    mla_gens = generate_samples(
        mla_model, tokenizer, prompts,
        max_new_tokens=60, temperature=0.7,
    )
    print("\n  MLA generations:")
    for p, g in zip(prompts, mla_gens):
        print(f"    > {p}")
        print(f"      {g[len(p):].strip()[:120]}")

    # ------------------------------------------------------------------
    # Step 8 – Orthonormality check
    # ------------------------------------------------------------------
    separator("Step 8: Orthonormality verification")

    handler = getattr(mla_model, "mla_handler", None)
    max_uk, max_uv = 0.0, 0.0

    for li in range(mla_model.mla_config.n_layers):
        if handler is not None:
            layer = handler.get_layer_module(li)
            attn = getattr(layer, handler.get_attention_attribute_name())
        elif mla_model.mla_config.model_type == "gpt2":
            attn = mla_model.transformer.h[li].attn
        else:
            attn = mla_model.model.layers[li].self_attn

        if hasattr(attn, "mla"):
            errs = attn.mla.check_orthonormality()
            max_uk = max(max_uk, errs["W_uk"][0])
            max_uv = max(max_uv, errs["W_uv"][0])

    print(f"  W_uk max error : {max_uk:.2e}")
    print(f"  W_uv max error : {max_uv:.2e}")
    print(f"  Status         : {'PRESERVED' if max(max_uk, max_uv) < 1e-3 else 'DRIFTED'}")

    # ------------------------------------------------------------------
    # Save trained model
    # ------------------------------------------------------------------
    separator("Saving trained model")
    save_mla_model(mla_model, tokenizer, TRAINED)

    # ------------------------------------------------------------------
    # Step 9 – HF AutoModel round-trip
    # ------------------------------------------------------------------
    separator("Step 9: HuggingFace AutoModel round-trip")

    if 'trainer' in dir():
        trainer.cleanup()
        del trainer
    # Clear all local references that keep the model alive
    if 'handler' in dir():
        del handler
    if 'attn' in dir():
        del attn
    del mla_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU mem after cleanup: {gpu_mem():.2f} GB")

    print(f"  Loading via AutoModelForCausalLM.from_pretrained('{TRAINED}') …")
    t0 = time.time()
    hf_model = AutoModelForCausalLM.from_pretrained(
        TRAINED, trust_remote_code=True, torch_dtype=DTYPE,
    )
    hf_tok = AutoTokenizer.from_pretrained(TRAINED)
    if hf_tok.pad_token is None:
        hf_tok.pad_token = hf_tok.eos_token
    print(f"  HF load time : {time.time()-t0:.1f}s")
    print(f"  GPU mem      : {gpu_mem():.2f} GB")

    print("\n  Generating a short essay on AI …")
    prompt = "Artificial intelligence will transform society by"
    inputs = hf_tok(prompt, return_tensors="pt").to(DEVICE)
    hf_model.generation_config.pad_token_id = hf_tok.eos_token_id
    with torch.no_grad():
        out = hf_model.generate(
            inputs.input_ids, max_new_tokens=150,
            temperature=0.7, do_sample=True,
        )
    text = hf_tok.decode(out[0], skip_special_tokens=True)
    print(f"\n  {text}\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    separator("BENCHMARK SUMMARY")
    use_xkv = getattr(mla_model if 'mla_model' in dir() else hf_model,
                       "mla_config", None)
    print(f"  Model              : {MODEL_NAME} (1.6B params)")
    print(f"  Handler            : GenericHandler (auto-discovered)")
    print(f"  model_type         : stablelm (no dedicated handler)")
    print(f"  Compression target : {COMPRESSION_RATIO}x")
    print(f"  Actual ratio       : {cfg['theoretical_compression_ratio']:.2f}x")
    print(f"  Baseline PPL       : {baseline_ppl:.2f}")
    if pre_train_ppl is not None:
        print(f"  Pre-train PPL      : {pre_train_ppl:.2f} (+{((pre_train_ppl/baseline_ppl)-1)*100:.1f}%)")
    print(f"  Post-train PPL     : {post_ppl:.2f} (+{((post_ppl/baseline_ppl)-1)*100:.1f}%)")
    print(f"  Orthonormality     : {'PRESERVED' if max(max_uk, max_uv) < 1e-3 else 'DRIFTED'} (max err {max(max_uk, max_uv):.2e})")
    print()


if __name__ == "__main__":
    main()
