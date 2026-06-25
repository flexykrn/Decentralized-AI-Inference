import math
from typing import List, Dict
from shared.models import Provider
from admin.calculator import calculate_model_memory

def match_providers_to_model(model_id: str, precision: str, context_length: int, providers: List[Provider]) -> dict:
    req = calculate_model_memory(model_id, precision, context_length)
    total_layers = req["layers"]
    memory_per_layer = req["memory_per_layer_gb"]
    
    # Sort by memory descending (greedy)
    sorted_providers = sorted(providers, key=lambda p: p.device_memory, reverse=True)
    
    assignments = []
    remaining_layers = total_layers
    current_layer = 0
    
    for provider in sorted_providers:
        if remaining_layers <= 0:
            break
        
        # Account for per-layer overhead
        kv_per_layer = req["kv_cache_gb"] / total_layers
        activation_per_layer = req["activation_gb"] / total_layers
        mem_per_layer_total = memory_per_layer + kv_per_layer + activation_per_layer
        
        layers_for_provider = math.floor(provider.device_memory / mem_per_layer_total)
        layers_for_provider = min(layers_for_provider, remaining_layers)
        
        if layers_for_provider > 0:
            start = current_layer
            end = current_layer + layers_for_provider - 1
            assignments.append({
                "provider_id": provider.device_id,
                "layers": (start, end),
                "layer_count": layers_for_provider,
                "memory_used_gb": round(layers_for_provider * mem_per_layer_total, 2),
                "memory_available_gb": provider.device_memory,
                "backend": provider.compute_backend,
                "os": provider.os_type,
            })
            current_layer += layers_for_provider
            remaining_layers -= layers_for_provider
    
    can_run = remaining_layers <= 0
    
    return {
        "can_run": can_run,
        "model_id": model_id,
        "precision": precision,
        "total_layers": total_layers,
        "memory_per_layer_gb": req["memory_per_layer_gb"],
        "total_memory_needed_gb": req["total_memory_gb"],
        "providers_available": len(providers),
        "providers_used": len(assignments),
        "assignments": assignments,
        "remaining_layers": remaining_layers if not can_run else 0,
    }