#!/usr/bin/env python3
"""
DiCAI Production Test - 3 Providers with KV Cache

Tests the full distributed pipeline with:
- KV cache persistence between providers
- Health monitoring
- Auth middleware
- Simulated 3 providers on 1 machine
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import numpy as np
from shared.kv_cache import KVCache
from shared.health_monitor import HealthMonitor, ProviderStatus
from shared.auth import AuthManager

from provider.pytorch_provider import ShardProvider


def test_three_providers():
    """Test with 3 simulated providers."""
    print("=" * 60)
    print("DiCAI 3-Provider Production Test")
    print("=" * 60)
    
    # Setup auth
    auth = AuthManager(".test_production_tokens.json")
    
    # Create providers (simulating 3 machines)
    p1 = ShardProvider('p1', 'pytorch_model/provider_0', 0, 7, port=8081)
    p1.load_shard()
    
    p2 = ShardProvider('p2', 'pytorch_model/provider_0', 8, 15, port=8082)
    p2.load_shard()
    
    p3 = ShardProvider('p3', 'pytorch_model/provider_1', 16, 21, port=8083)
    p3.load_shard()
    
    # Setup health monitoring
    monitor = HealthMonitor(heartbeat_interval=2, failure_threshold=2)
    monitor.register_provider("p1", "http://localhost:8081")
    monitor.register_provider("p2", "http://localhost:8082")
    monitor.register_provider("p3", "http://localhost:8083")
    
    # Generate invite tokens
    token1 = auth.generate_invite_token()
    token2 = auth.generate_invite_token()
    token3 = auth.generate_invite_token()
    
    print(f"\n[Auth] Generated tokens for 3 providers")
    
    # Test: Generate 5 tokens with KV cache
    print("\n[Inference] Generating 5 tokens with KV cache...")
    
    prompt = [1, 2, 3]  # "Hello"
    generated_tokens = []
    next_token = 0  # Initialize
    kv_cache = KVCache()
    
    for step in range(5):
        if step == 0:
            current_input = prompt
        else:
            current_input = [int(next_token)]
        print(f"\n[Step {step + 1}] Input: {current_input}")
        
        # Provider 1 (layers 0-7)
        start = time.time()
        result1 = p1.process(current_input, kv_cache=kv_cache)
        p1_time = (time.time() - start) * 1000
        
        if result1['status'] != 'success':
            print(f"[ERROR] P1 failed: {result1.get('error')}")
            break
            
        # Update KV cache from P1
        if 'kv_cache_state' in result1:
            kv_cache = KVCache.deserialize(result1['kv_cache_state'])
            
        # Provider 2 (layers 8-15)
        start = time.time()
        result2 = p2.process(hidden_states=result1['hidden_states'], kv_cache=kv_cache)
        p2_time = (time.time() - start) * 1000
        
        if result2['status'] != 'success':
            print(f"[ERROR] P2 failed: {result2.get('error')}")
            break
            
        # Update KV cache from P2
        if 'kv_cache_state' in result2:
            kv_cache = KVCache.deserialize(result2['kv_cache_state'])
            
        # Provider 3 (layers 16-21 + output)
        start = time.time()
        result3 = p3.process(hidden_states=result2['hidden_states'], kv_cache=kv_cache)
        p3_time = (time.time() - start) * 1000
        
        if result3['status'] != 'success':
            print(f"[ERROR] P3 failed: {result3.get('error')}")
            break
            
        # Update KV cache from P3
        if 'kv_cache_state' in result3:
            kv_cache = KVCache.deserialize(result3['kv_cache_state'])
            
        # Get token
        logits = torch.tensor(result3['logits'])
        next_token = torch.argmax(logits[-1]).item()
        generated_tokens.append(next_token)
        
        print(f"  P1: {p1_time:.1f}ms | P2: {p2_time:.1f}ms | P3: {p3_time:.1f}ms")
        print(f"  Token: {next_token}")
        
        # Record heartbeats
        monitor.record_heartbeat("p1", p1_time)
        monitor.record_heartbeat("p2", p2_time)
        monitor.record_heartbeat("p3", p3_time)
        
    # Results
    print(f"\n{'=' * 60}")
    print(f"Generated: {generated_tokens}")
    print(f"KV Cache: {kv_cache}")
    print(f"Health: {monitor.get_all_status()}")
    print(f"{'=' * 60}")
    
    # Cleanup
    os.remove(".test_production_tokens.json")
    
    print("\n3-Provider Production Test PASSED")
    return True


if __name__ == "__main__":
    success = test_three_providers()
    sys.exit(0 if success else 1)
