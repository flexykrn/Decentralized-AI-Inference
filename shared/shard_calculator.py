"""
Model Shard Calculator

For distributed inference, we split transformer layers across providers.
Each provider loads only its assigned layers.

Architecture for 1T model:
- ~100-200 layers (depending on model design)
- Each layer: ~5GB at Q4_K_M
- Provider with 16GB RAM: can hold ~3 layers
- Need ~67 providers for 200 layers

For smaller models (testing):
- 1.1B model: ~22 layers, ~0.5GB total
- Can test sharding on single machine with multiple ports
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class ModelConfig:
    name: str
    total_layers: int
    total_params: int  # in billions
    memory_per_layer_q4km: float  # in GB
    memory_per_layer_bf16: float  # in GB
    embedding_memory: float  # in GB
    
    def total_memory_gb(self, precision: str = "q4_k_m") -> float:
        """Total model memory."""
        if precision == "q4_k_m":
            layer_mem = self.memory_per_layer_q4km
        elif precision == "bf16":
            layer_mem = self.memory_per_layer_bf16
        else:
            layer_mem = self.memory_per_layer_q4km
        
        return (self.total_layers * layer_mem) + self.embedding_memory


# Known model configs
MODELS = {
    "tinyllama-1.1b": ModelConfig(
        name="TinyLlama 1.1B",
        total_layers=22,
        total_params=1,
        memory_per_layer_q4km=0.02,  # ~20MB per layer
        memory_per_layer_bf16=0.08,
        embedding_memory=0.1,
    ),
    "qwen2.5-7b": ModelConfig(
        name="Qwen2.5 7B",
        total_layers=28,
        total_params=7,
        memory_per_layer_q4km=0.15,
        memory_per_layer_bf16=0.6,
        embedding_memory=0.2,
    ),
    "llama-3-70b": ModelConfig(
        name="Llama 3 70B",
        total_layers=80,
        total_params=70,
        memory_per_layer_q4km=0.37,
        memory_per_layer_bf16=1.63,
        embedding_memory=0.5,
    ),
    "llama-3-405b": ModelConfig(
        name="Llama 3 405B",
        total_layers=126,
        total_params=405,
        memory_per_layer_q4km=2.1,
        memory_per_layer_bf16=8.4,
        embedding_memory=2.0,
    ),
    "future-1t": ModelConfig(
        name="Future 1T Model",
        total_layers=200,
        total_params=1000,
        memory_per_layer_q4km=5.0,
        memory_per_layer_bf16=20.0,
        embedding_memory=5.0,
    ),
}


def calculate_shards(
    model_name: str,
    precision: str,
    providers: List[Dict]
) -> List[Dict]:
    """
    Calculate layer assignments for each provider.
    
    Returns list of assignments: [{provider_id, layers: [start, end], memory_gb}]
    """
    model = MODELS.get(model_name)
    if not model:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Sort providers by available memory (descending)
    sorted_providers = sorted(
        providers,
        key=lambda p: p.get("memory_gb", 16),
        reverse=True
    )
    
    if precision == "q4_k_m":
        layer_mem = model.memory_per_layer_q4km
    elif precision == "bf16":
        layer_mem = model.memory_per_layer_bf16
    else:
        layer_mem = model.memory_per_layer_q4km
    
    assignments = []
    current_layer = 0
    
    for provider in sorted_providers:
        if current_layer >= model.total_layers:
            break
        
        available_mem = provider.get("memory_gb", 16)
        # Reserve 2GB for overhead
        usable_mem = max(available_mem - 2, 2)
        
        # How many layers can this provider hold?
        max_layers = int(usable_mem / layer_mem)
        
        # Don't assign more layers than remaining
        layers_to_assign = min(max_layers, model.total_layers - current_layer)
        
        if layers_to_assign > 0:
            start_layer = current_layer
            end_layer = current_layer + layers_to_assign - 1
            
            assignments.append({
                "provider_id": provider["id"],
                "provider_address": provider.get("address", "localhost"),
                "provider_port": provider.get("port", 8081),
                "layers": [start_layer, end_layer],
                "layer_count": layers_to_assign,
                "memory_required_gb": layers_to_assign * layer_mem + model.embedding_memory / len([p for p in sorted_providers if p.get("memory_gb", 16) > 2]),
                "model_name": model_name,
                "precision": precision,
            })
            
            current_layer += layers_to_assign
    
    # Check if all layers assigned
    if current_layer < model.total_layers:
        return {
            "error": f"Insufficient providers. Only assigned {current_layer}/{model.total_layers} layers",
            "assignments": assignments,
            "needed_more": model.total_layers - current_layer,
        }
    
    return {
        "model": model.name,
        "total_layers": model.total_layers,
        "precision": precision,
        "total_memory_gb": model.total_memory_gb(precision),
        "assignments": assignments,
        "provider_count": len(assignments),
    }


def print_shard_plan(plan: dict):
    """Pretty print shard plan."""
    if "error" in plan:
        print(f"ERROR: {plan['error']}")
        print(f"Need {plan['needed_more']} more layers worth of memory")
        return
    
    print(f"\n{'='*60}")
    print(f"Model: {plan['model']}")
    print(f"Precision: {plan['precision']}")
    print(f"Total layers: {plan['total_layers']}")
    print(f"Total memory: {plan['total_memory_gb']:.1f} GB")
    print(f"Providers needed: {plan['provider_count']}")
    print(f"{'='*60}")
    
    for a in plan['assignments']:
        print(f"\nProvider: {a['provider_id']} ({a['provider_address']}:{a['provider_port']})")
        print(f"  Layers: {a['layers'][0]}-{a['layers'][1]} ({a['layer_count']} layers)")
        print(f"  Memory: {a['memory_required_gb']:.2f} GB")


if __name__ == "__main__":
    # Test with 1T model
    providers_1t = [
        {"id": f"p{i}", "memory_gb": 16, "address": "localhost", "port": 8081+i}
        for i in range(100)
    ]
    
    plan = calculate_shards("future-1t", "q4_k_m", providers_1t)
    print_shard_plan(plan)
    
    # Test with TinyLlama (for local testing)
    print("\n" + "="*60)
    print("LOCAL TEST CONFIG")
    print("="*60)
    providers_test = [
        {"id": f"p{i}", "memory_gb": 8, "address": "localhost", "port": 8081+i}
        for i in range(5)
    ]
    
    plan_test = calculate_shards("tinyllama-1.1b", "q4_k_m", providers_test)
    print_shard_plan(plan_test)
