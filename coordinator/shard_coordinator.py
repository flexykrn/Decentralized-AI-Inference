#!/usr/bin/env python3
"""
DiCAI Custom Sharding Protocol - Coordinator

Manages distributed inference across multiple providers.
Each provider holds a shard (subset of layers).
Activations flow sequentially through the chain.
"""

import argparse
import json
import time
from typing import List, Dict, Optional
import urllib.request
from dataclasses import dataclass


@dataclass
class ProviderShard:
    provider_id: str
    address: str
    port: int
    start_layer: int
    end_layer: int


class DistributedInferenceCoordinator:
    """
    Coordinator that routes inference requests through a chain of providers.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.providers: List[ProviderShard] = []
        
    def register_provider(self, provider_id: str, address: str, port: int, 
                          start_layer: int, end_layer: int):
        """Register a provider with its layer shard."""
        shard = ProviderShard(provider_id, address, port, start_layer, end_layer)
        self.providers.append(shard)
        print(f"[Coordinator] Registered {provider_id} at {address}:{port} (layers {start_layer}-{end_layer})")
        
    def get_provider_chain(self) -> List[ProviderShard]:
        """Get providers sorted by layer order."""
        return sorted(self.providers, key=lambda p: p.start_layer)
    
    def health_check_all(self) -> Dict:
        """Check health of all providers."""
        results = {}
        for provider in self.providers:
            try:
                req = urllib.request.Request(
                    f"http://{provider.address}:{provider.port}/health",
                    method="GET"
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    results[provider.provider_id] = json.loads(resp.read())
            except Exception as e:
                results[provider.provider_id] = {"error": str(e)}
        return results
    
    def run_inference(self, prompt: str, max_tokens: int = 20) -> str:
        """
        Run distributed inference through the provider chain.
        
        For true sharding, this would:
        1. Tokenize prompt
        2. Send to first provider
        3. Get output activations
        4. Send to next provider
        5. Repeat until last provider
        6. Apply LM head, sample next token
        7. Repeat for max_tokens
        
        Current implementation: route to first provider for simplicity.
        """
        chain = self.get_provider_chain()
        if not chain:
            raise RuntimeError("No providers registered")
        
        # For testing, send to first provider
        # In production, implement full chain routing
        first_provider = chain[0]
        
        try:
            data = json.dumps({
                "input_ids": [1, 2, 3],  # Dummy token IDs
                "past_kv": None
            }).encode()
            
            req = urllib.request.Request(
                f"http://{first_provider.address}:{first_provider.port}/process",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                return result.get("output_text", "")
                
        except Exception as e:
            return f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="DiCAI Distributed Inference Coordinator")
    parser.add_argument("--port", type=int, default=8080, help="Coordinator port")
    args = parser.parse_args()
    
    coordinator = DistributedInferenceCoordinator("tinyllama-1.1b")
    
    # Register providers (in production, these would register themselves)
    coordinator.register_provider("p1", "localhost", 8081, 0, 10)
    coordinator.register_provider("p2", "localhost", 8082, 11, 21)
    
    print(f"="*60)
    print(f"DiCAI Distributed Inference Coordinator")
    print(f"="*60)
    print(f"Port: {args.port}")
    print(f"Providers: {len(coordinator.providers)}")
    print(f"="*60)
    
    # Health check
    print("\nHealth checking providers...")
    health = coordinator.health_check_all()
    for pid, status in health.items():
        print(f"  {pid}: {status.get('status', 'unknown')}")
    
    # Test inference
    print("\nTesting inference...")
    result = coordinator.run_inference("Hello, I am")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
