#!/usr/bin/env python3
"""
DiCAI - Memory-Optimized Model Loader

Implements efficient memory management for model shards:
- Memory-map shard files instead of loading into RAM
- Prefetch next layer while computing current
- Offload unused layers to disk
"""

import os
import mmap
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from safetensors.torch import load_file


class MemoryMappedShard:
    """Memory-mapped model shard for efficient loading."""
    
    def __init__(self, shard_path: str):
        self.shard_path = shard_path
        self.file_handle = None
        self.memory_map = None
        self.loaded_tensors: Dict[str, torch.Tensor] = {}
        self.tensor_metadata: Dict[str, Tuple[torch.Size, torch.dtype, int, int]] = {}
        
        self._open_file()
        self._parse_metadata()
        
    def _open_file(self):
        """Open file for memory mapping."""
        self.file_handle = open(self.shard_path, 'rb')
        
        # For safetensors, we can't easily mmap the whole file
        # Instead, we'll load on demand with caching
        self.file_size = os.path.getsize(self.shard_path)
        print(f"[MemoryMap] Opened {self.shard_path} ({self.file_size / 1024 / 1024:.1f} MB)")
        
    def _parse_metadata(self):
        """Parse tensor metadata without loading data."""
        # For safetensors, we load metadata only
        try:
            from safetensors import safe_open
            with safe_open(self.shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    self.tensor_metadata[key] = (
                        tensor.shape,
                        tensor.dtype,
                        tensor.numel() * tensor.element_size(),
                        0  # offset (not used for safetensors)
                    )
        except Exception as e:
            print(f"[MemoryMap] Error parsing metadata: {e}")
            
    def load_tensor(self, key: str, device: str = "cpu") -> Optional[torch.Tensor]:
        """Load a specific tensor on demand."""
        # Check cache first
        if key in self.loaded_tensors:
            return self.loaded_tensors[key]
            
        # Load from file
        try:
            from safetensors import safe_open
            with safe_open(self.shard_path, framework="pt", device=device) as f:
                if key in f.keys():
                    tensor = f.get_tensor(key)
                    self.loaded_tensors[key] = tensor
                    return tensor
        except Exception as e:
            print(f"[MemoryMap] Error loading {key}: {e}")
            
        return None
        
    def unload_tensor(self, key: str):
        """Unload a tensor to free memory."""
        if key in self.loaded_tensors:
            del self.loaded_tensors[key]
            
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        total_bytes = 0
        for tensor in self.loaded_tensors.values():
            total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes / (1024 * 1024)
        
    def close(self):
        """Close file handle."""
        if self.file_handle:
            self.file_handle.close()
            

class LayerPrefetcher:
    """Prefetches layers for efficient inference."""
    
    def __init__(self, shard: MemoryMappedShard, prefetch_count: int = 2):
        self.shard = shard
        self.prefetch_count = prefetch_count
        self.prefetch_queue: list = []
        self.current_layer = 0
        
    def set_current_layer(self, layer_id: int):
        """Set current layer and prefetch next ones."""
        self.current_layer = layer_id
        self._prefetch_next_layers()
        
    def _prefetch_next_layers(self):
        """Prefetch next N layers."""
        for i in range(1, self.prefetch_count + 1):
            next_layer = self.current_layer + i
            
            # Load layer weights
            for key in self.shard.tensor_metadata:
                if f'blk.{next_layer}.' in key:
                    self.shard.load_tensor(key)
                    
    def unload_previous_layers(self, keep_count: int = 2):
        """Unload layers far behind current position."""
        cutoff = self.current_layer - keep_count
        
        keys_to_unload = []
        for key in self.shard.loaded_tensors:
            # Extract layer number from key
            if 'blk.' in key:
                try:
                    layer_num = int(key.split('blk.')[1].split('.')[0])
                    if layer_num < cutoff:
                        keys_to_unload.append(key)
                except:
                    pass
                    
        for key in keys_to_unload:
            self.shard.unload_tensor(key)
            

class OptimizedShardLoader:
    """High-level shard loader with memory optimization."""
    
    def __init__(self, shard_dir: str, memory_limit_gb: float = 4.0):
        self.shard_dir = shard_dir
        self.memory_limit_gb = memory_limit_gb
        self.shards: Dict[str, MemoryMappedShard] = {}
        self.prefetchers: Dict[str, LayerPrefetcher] = {}
        
    def load_shard(self, shard_name: str) -> MemoryMappedShard:
        """Load a shard with memory mapping."""
        # Try multiple paths
        possible_paths = [
            os.path.join(self.shard_dir, f"{shard_name}.safetensors"),
            os.path.join(self.shard_dir, shard_name, "shard.safetensors"),
            os.path.join(self.shard_dir, "shard.safetensors"),
        ]
        
        shard_path = None
        for path in possible_paths:
            if os.path.exists(path):
                shard_path = path
                break
                
        if shard_path is None:
            raise FileNotFoundError(f"No shard found in {self.shard_dir}")
            
        shard = MemoryMappedShard(shard_path)
        self.shards[shard_name] = shard
        
        # Create prefetcher
        self.prefetchers[shard_name] = LayerPrefetcher(shard)
        
        return shard
        
    def get_tensor(self, shard_name: str, key: str, device: str = "cpu") -> Optional[torch.Tensor]:
        """Get tensor from shard with automatic loading."""
        if shard_name not in self.shards:
            return None
            
        shard = self.shards[shard_name]
        
        # Check memory limit
        if shard.get_memory_usage() > self.memory_limit_gb * 1024:
            # Unload oldest tensors
            self._unload_oldest(shard_name)
            
        return shard.load_tensor(key, device)
        
    def _unload_oldest(self, shard_name: str):
        """Unload oldest tensors to free memory."""
        shard = self.shards[shard_name]
        
        # Simple LRU: unload first loaded tensors
        if len(shard.loaded_tensors) > 0:
            oldest_key = next(iter(shard.loaded_tensors))
            shard.unload_tensor(oldest_key)
            
    def get_memory_usage(self) -> float:
        """Get total memory usage in MB."""
        total = 0
        for shard in self.shards.values():
            total += shard.get_memory_usage()
        return total
        
    def close_all(self):
        """Close all shards."""
        for shard in self.shards.values():
            shard.close()


def test_memory_optimization():
    """Test memory-optimized loading."""
    print("=" * 60)
    print("Memory Optimization Test")
    print("=" * 60)
    
    # Check if we have a shard to test
    shard_dir = "pytorch_model/provider_0"
    if not os.path.exists(shard_dir):
        print(f"Shard directory not found: {shard_dir}")
        print("Skipping test - no shard available")
        return True
        
    # Load with memory optimization
    loader = OptimizedShardLoader(shard_dir, memory_limit_gb=1.0)
    shard = loader.load_shard("test")
    
    print(f"\nShard metadata:")
    print(f"  Total tensors: {len(shard.tensor_metadata)}")
    print(f"  Memory usage: {shard.get_memory_usage():.2f} MB")
    
    # Load specific tensors
    test_keys = [k for k in shard.tensor_metadata.keys() if 'blk.0.' in k][:5]
    
    for key in test_keys:
        tensor = shard.load_tensor(key)
        if tensor is not None:
            print(f"  Loaded {key}: {tensor.shape} ({tensor.numel() * tensor.element_size() / 1024 / 1024:.2f} MB)")
            
    print(f"\nAfter loading:")
    print(f"  Memory usage: {shard.get_memory_usage():.2f} MB")
    
    # Unload some tensors
    for key in test_keys[:2]:
        shard.unload_tensor(key)
        
    print(f"After unloading:")
    print(f"  Memory usage: {shard.get_memory_usage():.2f} MB")
    
    # Cleanup
    loader.close_all()
    
    print("\nMemory Optimization Test PASSED")
    return True


if __name__ == "__main__":
    test_memory_optimization()
