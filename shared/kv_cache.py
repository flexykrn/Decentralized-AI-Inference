#!/usr/bin/env python3
"""
DiCAI KV Cache

Persistent key-value cache for efficient multi-token inference.
Stores K,V tensors per layer to avoid recomputation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np


class KVCache:
    """Key-Value cache for transformer attention."""
    
    def __init__(self, max_seq_len: int = 4096, device: str = "cpu"):
        self.max_seq_len = max_seq_len
        self.device = device
        self.k_cache: Dict[int, torch.Tensor] = {}  # layer_id -> [batch, heads, seq, head_dim]
        self.v_cache: Dict[int, torch.Tensor] = {}  # layer_id -> [batch, heads, seq, head_dim]
        self.current_len = 0
        
    def init_cache(self, layer_id: int, batch_size: int, num_heads: int, head_dim: int):
        """Initialize empty cache for a layer."""
        if layer_id not in self.k_cache:
            self.k_cache[layer_id] = torch.zeros(
                batch_size, num_heads, 0, head_dim,
                device=self.device, dtype=torch.float32
            )
            self.v_cache[layer_id] = torch.zeros(
                batch_size, num_heads, 0, head_dim,
                device=self.device, dtype=torch.float32
            )
            
    def append(self, layer_id: int, k: torch.Tensor, v: torch.Tensor):
        """Append new K,V to cache."""
        if layer_id not in self.k_cache:
            self.k_cache[layer_id] = k
            self.v_cache[layer_id] = v
        else:
            # Concatenate along sequence dimension
            self.k_cache[layer_id] = torch.cat([self.k_cache[layer_id], k], dim=2)
            self.v_cache[layer_id] = torch.cat([self.v_cache[layer_id], v], dim=2)
            
        # Enforce max sequence length
        if self.k_cache[layer_id].shape[2] > self.max_seq_len:
            self.k_cache[layer_id] = self.k_cache[layer_id][:, :, -self.max_seq_len:, :]
            self.v_cache[layer_id] = self.v_cache[layer_id][:, :, -self.max_seq_len:, :]
            
        self.current_len = self.k_cache[layer_id].shape[2]
        
    def get(self, layer_id: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached K,V for a layer."""
        return self.k_cache.get(layer_id), self.v_cache.get(layer_id)
        
    def clear(self):
        """Clear all caches."""
        self.k_cache.clear()
        self.v_cache.clear()
        self.current_len = 0
        
    def get_cache_len(self) -> int:
        """Get current cache length."""
        return self.current_len
        
    def get_memory_usage(self) -> dict:
        """Get memory usage statistics."""
        total_bytes = 0
        for layer_id in self.k_cache:
            k_bytes = self.k_cache[layer_id].element_size() * self.k_cache[layer_id].nelement()
            v_bytes = self.v_cache[layer_id].element_size() * self.v_cache[layer_id].nelement()
            total_bytes += k_bytes + v_bytes
            
        return {
            "total_mb": total_bytes / (1024 * 1024),
            "layers": len(self.k_cache),
            "seq_len": self.current_len
        }
        
    def serialize(self) -> Dict[str, Any]:
        """Serialize cache to dict for network transfer."""
        return {
            'k': {str(k): v.detach().cpu().numpy().tolist() for k, v in self.k_cache.items()},
            'v': {str(k): v.detach().cpu().numpy().tolist() for k, v in self.v_cache.items()},
            'len': self.current_len
        }
        
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'KVCache':
        """Deserialize cache from dict."""
        cache = cls()
        cache.k_cache = {int(k): torch.tensor(v) for k, v in data.get('k', {}).items()}
        cache.v_cache = {int(k): torch.tensor(v) for k, v in data.get('v', {}).items()}
        cache.current_len = data.get('len', 0)
        return cache
        
    def __repr__(self):
        return f"KVCache(layers={len(self.k_cache)}, seq_len={self.current_len})"


class KVCacheManager:
    """Manages KV caches across multiple providers."""
    
    def __init__(self, max_seq_len: int = 4096):
        self.max_seq_len = max_seq_len
        self.caches: Dict[str, KVCache] = {}  # request_id -> KVCache
        
    def get_cache(self, request_id: str) -> KVCache:
        """Get or create cache for a request."""
        if request_id not in self.caches:
            self.caches[request_id] = KVCache(max_seq_len=self.max_seq_len)
        return self.caches[request_id]
        
    def clear_cache(self, request_id: str):
        """Clear cache for a request."""
        if request_id in self.caches:
            self.caches[request_id].clear()
            del self.caches[request_id]
            
    def clear_all(self):
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        self.caches.clear()
        
    def get_memory_usage(self) -> dict:
        """Get total memory usage across all caches."""
        total_mb = 0
        for request_id, cache in self.caches.items():
            usage = cache.get_memory_usage()
            total_mb += usage["total_mb"]
            
        return {
            "total_mb": total_mb,
            "num_requests": len(self.caches)
        }


def test_kv_cache():
    """Test KV cache functionality."""
    cache = KVCache(max_seq_len=100)
    
    # Simulate adding K,V for layer 0
    batch_size, num_heads, seq_len, head_dim = 1, 8, 1, 256
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    cache.append(0, k, v)
    print(f"Cache len after first append: {cache.get_cache_len()}")
    
    # Append more
    k2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
    cache.append(0, k2, v2)
    print(f"Cache len after second append: {cache.get_cache_len()}")
    
    # Get cache
    k_cached, v_cached = cache.get(0)
    print(f"Cached K shape: {k_cached.shape}")
    print(f"Cached V shape: {v_cached.shape}")
    
    # Memory usage
    usage = cache.get_memory_usage()
    print(f"Memory usage: {usage['total_mb']:.2f} MB")
    
    print("KV Cache test PASSED")


if __name__ == "__main__":
    test_kv_cache()
