import math
from shared.constants import MODEL_METADATA, PRECISION_MULTIPLIERS

def calculate_model_memory(model_id: str, precision: str, context_length: int = 4096) -> dict:
    if model_id not in MODEL_METADATA:
        raise ValueError(f"Unknown model: {model_id}")
    
    meta = MODEL_METADATA[model_id]
    params = meta["params_b"] * 1e9
    layers = meta["layers"]
    hidden_size = meta["hidden_size"]
    heads = meta["heads"]
    
    bytes_per_param = PRECISION_MULTIPLIERS.get(precision, 2.0)
    
    # Weights memory
    weights_memory_gb = (params * bytes_per_param) / (1024**3)
    memory_per_layer_gb = weights_memory_gb / layers
    
    # KV cache: 2 * layers * heads * head_dim * context_length * bytes_per_param
    head_dim = hidden_size // heads
    kv_cache_gb = (2 * layers * heads * head_dim * context_length * bytes_per_param) / (1024**3)
    
    # Activations (batch_size=1)
    activation_gb = (hidden_size * context_length * bytes_per_param * 4) / (1024**3)
    
    # Overhead (20%)
    overhead = 1.2
    total_memory_gb = (weights_memory_gb + kv_cache_gb + activation_gb) * overhead
    
    return {
        "weights_memory_gb": round(weights_memory_gb, 2),
        "memory_per_layer_gb": round(memory_per_layer_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "activation_gb": round(activation_gb, 2),
        "total_memory_gb": round(total_memory_gb, 2),
        "layers": layers,
        "params_b": meta["params_b"],
        "precision": precision,
        "context_length": context_length,
    }