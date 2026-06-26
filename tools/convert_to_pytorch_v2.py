#!/usr/bin/env python3
"""
DiCAI GGUF to PyTorch Converter - Proper Implementation

Uses the gguf library to properly dequantize weights.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from safetensors.torch import save_file

import gguf


class GGUFConverter:
    """Converts GGUF models to PyTorch format for true sharding."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.reader = None
        
    def load_gguf(self):
        """Load GGUF file using proper library."""
        print(f"[GGUF] Loading {self.model_path}")
        self.reader = gguf.GGUFReader(self.model_path)
        print(f"[GGUF] Loaded {len(self.reader.tensors)} tensors")
        
    def convert_to_pytorch(self, output_dir: str):
        """Convert GGUF to PyTorch format with proper dequantization."""
        print(f"[Convert] Converting to PyTorch format...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        pytorch_state = {}
        
        for tensor in self.reader.tensors:
            name = tensor.name
            print(f"[Convert] Processing {name}...")
            
            # Get dequantized data
            data = gguf.dequantize(tensor.data, tensor.tensor_type)
            
            # Convert to PyTorch tensor
            if isinstance(data, np.ndarray):
                pytorch_state[name] = torch.from_numpy(data.copy())
            else:
                # Handle other types
                pytorch_state[name] = torch.tensor(data)
                
        # Save metadata
        metadata = {
            'original_format': 'gguf',
            'tensor_count': len(pytorch_state),
            'architecture': 'llama'
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save using safetensors
        save_file(pytorch_state, os.path.join(output_dir, 'model.safetensors'))
        
        print(f"[Convert] Saved {len(pytorch_state)} tensors to {output_dir}")
        
        return pytorch_state
        
    def split_for_providers(self, num_providers: int, output_dir: str):
        """Split model into provider shards."""
        print(f"[Split] Splitting for {num_providers} providers")
        
        # First convert to PyTorch
        pytorch_state = self.convert_to_pytorch(output_dir)
        
        # Identify layers
        layers = {}
        for name in pytorch_state.keys():
            if 'blk.' in name:
                # Extract layer number
                parts = name.split('.')
                layer_num = int(parts[1])
                if layer_num not in layers:
                    layers[layer_num] = []
                layers[layer_num].append(name)
            elif 'token_embd' in name:
                if -1 not in layers:
                    layers[-1] = []
                layers[-1].append(name)
            elif name == 'output_norm.weight' or name == 'output.weight':
                if 999 not in layers:
                    layers[999] = []
                layers[999].append(name)
                
        # Get actual layer numbers
        layer_nums = sorted([k for k in layers.keys() if k >= 0 and k < 999])
        
        if not layer_nums:
            print("[Split] ERROR: No layers found!")
            return
            
        total_layers = len(layer_nums)
        layers_per_provider = total_layers // num_providers
        remainder = total_layers % num_providers
        
        print(f"[Split] Total layers: {total_layers}")
        print(f"[Split] Layers per provider: {layers_per_provider}")
        
        # Create shards
        start_idx = 0
        for i in range(num_providers):
            extra = 1 if i < remainder else 0
            end_idx = start_idx + layers_per_provider + extra - 1
            
            if start_idx < len(layer_nums):
                start_layer = layer_nums[start_idx]
                end_layer = layer_nums[min(end_idx, len(layer_nums) - 1)]
                
                shard_dir = os.path.join(output_dir, f'provider_{i}')
                os.makedirs(shard_dir, exist_ok=True)
                
                # Collect tensors for this shard
                shard_tensors = {}
                
                # Always include embeddings
                if -1 in layers:
                    for name in layers[-1]:
                        shard_tensors[name] = pytorch_state[name]
                        
                # Include requested layers
                for layer_num in range(start_layer, end_layer + 1):
                    if layer_num in layers:
                        for name in layers[layer_num]:
                            if name in pytorch_state:
                                shard_tensors[name] = pytorch_state[name]
                                
                # Include output layers ONLY if this is the last provider
                if i == num_providers - 1:
                    if 999 in layers:
                        for name in layers[999]:
                            if name in pytorch_state:
                                shard_tensors[name] = pytorch_state[name]
                                print(f"[Split] Including output layer: {name}")
                
                # Save shard
                save_file(shard_tensors, os.path.join(shard_dir, 'shard.safetensors'))
                
                # Save shard info
                total_size = sum(v.numel() * v.element_size() for v in shard_tensors.values())
                shard_info = {
                    'provider_id': i,
                    'start_layer': start_layer,
                    'end_layer': end_layer,
                    'tensor_count': len(shard_tensors),
                    'total_size_mb': total_size / (1024 * 1024)
                }
                
                with open(os.path.join(shard_dir, 'info.json'), 'w') as f:
                    json.dump(shard_info, f, indent=2)
                    
                print(f"[Split] Provider {i}: layers {start_layer}-{end_layer}, {len(shard_tensors)} tensors, {shard_info['total_size_mb']:.1f} MB")
                
                start_idx = end_idx + 1
                
        print(f"[Split] All shards saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DiCAI GGUF to PyTorch Converter")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--output-dir", default="pytorch_model", help="Output directory")
    parser.add_argument("--providers", type=int, default=2, help="Number of providers")
    
    args = parser.parse_args()
    
    converter = GGUFConverter(args.model)
    converter.load_gguf()
    converter.split_for_providers(args.providers, args.output_dir)
    
    
if __name__ == "__main__":
    main()
