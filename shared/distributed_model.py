"""
Distributed Model Sharding and Inference

Adapted from AirLLM's layer-by-layer approach for distributed execution.

Architecture:
- Admin splits model into layer shards
- Each provider downloads ONLY its assigned layers
- During inference: activations flow sequentially through providers
- Provider 1 (layers 0-9) -> Provider 2 (layers 10-19) -> ... -> Provider N

For 1T model:
- 200 layers, ~5GB per layer at Q4_K_M
- 16GB provider: holds ~3 layers
- Need ~67 providers
- Each request: round-trip through all providers (latency adds up)
"""

import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
# import numpy as np  # optional, not needed for core logic


@dataclass
class LayerShard:
    """Represents a shard of model layers assigned to one provider."""
    provider_id: str
    provider_address: str
    provider_port: int
    start_layer: int
    end_layer: int
    layer_count: int
    memory_gb: float
    shard_file: str  # Path to the shard file (or URL to download)


class DistributedModelManager:
    """
    Manages distributed model execution across multiple providers.
    
    Similar to AirLLM but distributed:
    - Each provider loads its layer shard and keeps it in memory
    - Coordinator routes requests through providers sequentially
    - Activations (hidden states) passed between providers via HTTP
    """
    
    def __init__(self, model_name: str, precision: str = "q4_k_m"):
        self.model_name = model_name
        self.precision = precision
        self.shards: List[LayerShard] = []
        self.providers: Dict[str, dict] = {}
        
    def register_provider(self, provider_id: str, address: str, port: int, memory_gb: float):
        """Register a provider node."""
        self.providers[provider_id] = {
            "id": provider_id,
            "address": address,
            "port": port,
            "memory_gb": memory_gb,
            "status": "idle",
            "current_shard": None,
        }
        
    def calculate_shards(self, total_layers: int, memory_per_layer_gb: float) -> List[LayerShard]:
        """
        Calculate optimal layer distribution across providers.
        
        AirLLM-style: contiguous layer blocks per provider.
        """
        # Sort providers by memory
        sorted_providers = sorted(
            self.providers.values(),
            key=lambda p: p["memory_gb"],
            reverse=True
        )
        
        shards = []
        current_layer = 0
        
        for provider in sorted_providers:
            if current_layer >= total_layers:
                break
                
            # Reserve 2GB for OS/overhead, use 80% of remaining for layers
            usable_mem = max((provider["memory_gb"] - 2) * 0.8, 1)
            max_layers = int(usable_mem / memory_per_layer_gb)
            
            layers_to_assign = min(max_layers, total_layers - current_layer)
            
            if layers_to_assign > 0:
                shard = LayerShard(
                    provider_id=provider["id"],
                    provider_address=provider["address"],
                    provider_port=provider["port"],
                    start_layer=current_layer,
                    end_layer=current_layer + layers_to_assign - 1,
                    layer_count=layers_to_assign,
                    memory_gb=layers_to_assign * memory_per_layer_gb,
                    shard_file=f"{self.model_name}_layers_{current_layer}_{current_layer + layers_to_assign - 1}.gguf",
                )
                shards.append(shard)
                current_layer += layers_to_assign
                
        self.shards = shards
        return shards
    
    def get_inference_chain(self) -> List[LayerShard]:
        """Get ordered list of shards for inference (sorted by layer number)."""
        return sorted(self.shards, key=lambda s: s.start_layer)
    
    def estimate_latency(self, prompt_tokens: int, output_tokens: int) -> Dict:
        """
        Estimate inference latency.
        
        For distributed pipeline:
        - Each token must traverse ALL providers
        - Latency = num_providers * network_latency + computation_time
        """
        num_providers = len(self.shards)
        if num_providers == 0:
            return {"error": "No shards configured", "num_providers": 0}
        
        # Network latency between providers (assume 1-10ms over LAN)
        network_latency_ms = 5  # per hop
        
        # Computation time per layer (varies by hardware)
        # Assume ~10ms per layer on CPU, ~1ms on GPU
        comp_per_layer_ms = 10
        
        avg_layers_per_provider = sum(s.layer_count for s in self.shards) / num_providers
        comp_per_provider_ms = avg_layers_per_provider * comp_per_layer_ms
        
        # For prompt processing (parallelizable across layers to some extent)
        prompt_latency_ms = (
            prompt_tokens * 
            (comp_per_provider_ms + network_latency_ms) * 
            num_providers
        )
        
        # For token generation (sequential, must go through all providers per token)
        per_token_latency_ms = (
            comp_per_provider_ms * num_providers + 
            network_latency_ms * (num_providers - 1)
        )
        
        total_latency_ms = prompt_latency_ms + (output_tokens * per_token_latency_ms)
        
        return {
            "num_providers": num_providers,
            "prompt_processing_ms": prompt_latency_ms,
            "per_token_ms": per_token_latency_ms,
            "total_tokens": prompt_tokens + output_tokens,
            "estimated_total_ms": total_latency_ms,
            "estimated_total_seconds": total_latency_ms / 1000,
            "throughput_tok_per_sec": (prompt_tokens + output_tokens) / (total_latency_ms / 1000),
        }


# Model configurations (parameter count -> layer count, memory per layer)
MODEL_CONFIGS = {
    "tinyllama-1.1b": {"layers": 22, "mem_per_layer_q4km": 0.02, "mem_per_layer_bf16": 0.08},
    "qwen2.5-7b": {"layers": 28, "mem_per_layer_q4km": 0.15, "mem_per_layer_bf16": 0.6},
    "llama-3-8b": {"layers": 32, "mem_per_layer_q4km": 0.17, "mem_per_layer_bf16": 0.68},
    "llama-3-70b": {"layers": 80, "mem_per_layer_q4km": 0.37, "mem_per_layer_bf16": 1.63},
    "llama-3-405b": {"layers": 126, "mem_per_layer_q4km": 2.1, "mem_per_layer_bf16": 8.4},
    "future-1t": {"layers": 200, "mem_per_layer_q4km": 5.0, "mem_per_layer_bf16": 20.0},
}


def create_test_scenario(model_name: str, precision: str, provider_count: int, memory_per_provider: float):
    """Create a test scenario and show shard distribution."""
    config = MODEL_CONFIGS[model_name]
    
    manager = DistributedModelManager(model_name, precision)
    
    # Register providers
    for i in range(provider_count):
        manager.register_provider(
            f"p{i}", 
            f"localhost", 
            8081 + i, 
            memory_per_provider
        )
    
    # Calculate shards
    if precision == "q4_k_m":
        mem_per_layer = config["mem_per_layer_q4km"]
    else:
        mem_per_layer = config["mem_per_layer_bf16"]
    
    shards = manager.calculate_shards(config["layers"], mem_per_layer)
    
    print(f"\n{'='*70}")
    print(f"Model: {model_name} | Precision: {precision}")
    print(f"Total layers: {config['layers']} | Memory per layer: {mem_per_layer} GB")
    print(f"Providers: {provider_count} x {memory_per_provider} GB")
    print(f"{'='*70}")
    
    total_mem = 0
    for shard in shards:
        print(f"Provider {shard.provider_id}: layers {shard.start_layer}-{shard.end_layer} "
              f"({shard.layer_count} layers, {shard.memory_gb:.2f} GB)")
        total_mem += shard.memory_gb
    
    print(f"\nTotal memory across all providers: {total_mem:.1f} GB")
    
    # Estimate latency
    latency = manager.estimate_latency(prompt_tokens=50, output_tokens=100)
    print(f"\nEstimated latency for 50 prompt + 100 output tokens:")
    if "error" in latency:
        print(f"  ERROR: {latency['error']}")
    else:
        print(f"  Prompt processing: {latency['prompt_processing_ms']:.0f} ms")
        print(f"  Per token generation: {latency['per_token_ms']:.1f} ms")
        print(f"  Total: {latency['estimated_total_seconds']:.1f} seconds")
        print(f"  Throughput: {latency['throughput_tok_per_sec']:.1f} tokens/sec")
    
    return manager


if __name__ == "__main__":
    # Test scenarios
    print("\n" + "="*70)
    print("SCENARIO 1: 1T Model with 100 providers (16GB each)")
    print("="*70)
    create_test_scenario("future-1t", "q4_k_m", 100, 16)
    
    print("\n" + "="*70)
    print("SCENARIO 2: 1T Model BF16 (will fail - not enough memory)")
    print("="*70)
    create_test_scenario("future-1t", "bf16", 100, 16)
    
    print("\n" + "="*70)
    print("SCENARIO 3: 70B Model with 10 providers (16GB each)")
    print("="*70)
    create_test_scenario("llama-3-70b", "q4_k_m", 10, 16)
    
    print("\n" + "="*70)
    print("SCENARIO 4: Local test - TinyLlama with 3 providers (4GB each)")
    print("="*70)
    create_test_scenario("tinyllama-1.1b", "q4_k_m", 3, 4)
