MODEL_METADATA = {
    "llama-3-8b":   {"params_b": 8,   "layers": 32,  "hidden_size": 4096,  "heads": 32},
    "llama-3-70b":  {"params_b": 70,  "layers": 80,  "hidden_size": 8192,  "heads": 64},
    "llama-3-405b": {"params_b": 405, "layers": 126, "hidden_size": 16384, "heads": 128},
    "qwen-2.5-72b": {"params_b": 72,  "layers": 80,  "hidden_size": 8192,  "heads": 64},
    "mistral-7b":   {"params_b": 7,   "layers": 32,  "hidden_size": 4096,  "heads": 32},
    "deepseek-67b": {"params_b": 67,  "layers": 95,  "hidden_size": 8192,  "heads": 64},
}

PRECISION_MULTIPLIERS = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "q4_k_m": 0.45,
    "q5_k_m": 0.55,
    "q8_0": 1.0,
}
