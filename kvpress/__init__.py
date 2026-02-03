"""kvpress: KV Cache Compression via Multi-Head Latent Attention.

This library converts HuggingFace transformer models to use Multi-Head Latent
Attention (MLA) for KV cache compression. The compression matrices are
constrained to Stiefel manifolds (orthonormal columns) and optimized using
Riemannian gradient descent.

Example usage:

    from kvpress import convert_to_mla, save_mla_model, load_mla_model

    # Convert a model
    model, tokenizer = convert_to_mla("gpt2", compression_ratio=4)

    # Generate text
    inputs = tokenizer("Hello", return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_new_tokens=20)
    print(tokenizer.decode(outputs[0]))

    # Save and load
    save_mla_model(model, tokenizer, "my-mla-model")
    model, tokenizer = load_mla_model("my-mla-model")
"""

__version__ = "0.1.0"

# Core conversion function
from .converter import convert_to_mla

# Save/load functions
from .saving import save_mla_model, load_mla_model

# Training
from .trainer import MLATrainer, TrainingConfig

# Configuration
from .config import MLAConfig

# Evaluation
from .evaluation import (
    compute_perplexity,
    measure_cache_memory,
    compare_outputs,
    generate_samples,
)

# Core modules (for advanced usage)
from .compression import MLACompression
from .attention import MLAAttention, RotaryEmbedding, MLACache

# Initialization
from .initialization import (
    balanced_svd_init,
    collect_calibration_data,
    init_compression_from_calibration,
)

# Utilities
from .utils import (
    orthonormalize_rows,
    check_orthonormality,
    repeat_kv,
)

# Model handlers (for extending to new models)
from .model_handlers import (
    ModelHandler,
    GPT2Handler,
    LlamaHandler,
    MistralHandler,
    QwenHandler,
    get_handler,
    register_handler,
)

__all__ = [
    # Version
    "__version__",
    # Main API
    "convert_to_mla",
    "save_mla_model",
    "load_mla_model",
    "MLATrainer",
    "TrainingConfig",
    "MLAConfig",
    # Evaluation
    "compute_perplexity",
    "measure_cache_memory",
    "compare_outputs",
    "generate_samples",
    # Core modules
    "MLACompression",
    "MLAAttention",
    "RotaryEmbedding",
    "MLACache",
    # Initialization
    "balanced_svd_init",
    "collect_calibration_data",
    "init_compression_from_calibration",
    # Utilities
    "orthonormalize_rows",
    "check_orthonormality",
    "repeat_kv",
    # Model handlers
    "ModelHandler",
    "GPT2Handler",
    "LlamaHandler",
    "MistralHandler",
    "QwenHandler",
    "get_handler",
    "register_handler",
]
