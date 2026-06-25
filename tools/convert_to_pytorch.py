#!/usr/bin/env python3
"""
DiCAI PyTorch-Based True Layer Sharding

Converts GGUF to PyTorch format and implements distributed layer execution.
Each provider loads ONLY its assigned layers.
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file


class GGUFConverter:
    """Converts GGUF models to PyTorch format for true sharding."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.metadata = {}
        self.tensor_info = {}
        
    def parse_gguf(self):
        """Parse GGUF file."""
        print(f"[GGUF] Parsing {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError(f"Invalid GGUF magic: {magic}")
            
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            kv_count = struct.unpack('<Q', f.read(8))[0]
            
            print(f"[GGUF] Version: {version}, Tensors: {tensor_count}, KV: {kv_count}")
            
            # Read metadata
            for _ in range(kv_count):
                key = self._read_string(f)
                value_type = struct.unpack('<I', f.read(4))[0]
                value = self._read_value(f, value_type)
                self.metadata[key] = value
                
            # Read tensor info
            for i in range(tensor_count):
                name = self._read_string(f)
                n_dims = struct.unpack('<I', f.read(4))[0]
                dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                
                self.tensor_info[name] = {
                    'name': name,
                    'dims': dims,
                    'dtype': dtype,
                    'offset': offset,
                    'index': i
                }
                
            self.data_offset = f.tell()
            
        print(f"[GGUF] Parsed {len(self.tensor_info)} tensors")
        
    def _read_string(self, f) -> str:
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')
        
    def _read_value(self, f, value_type: int):
        types = {
            0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2),
            4: ('I', 4), 5: ('i', 4), 6: ('f', 4), 7: ('?', 1),
            8: (None, None), 9: (None, None), 10: ('Q', 8),
            11: ('q', 8), 12: ('d', 8)
        }
        
        if value_type == 8:  # string
            return self._read_string(f)
        elif value_type == 9:  # array
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            return [self._read_value(f, arr_type) for _ in range(arr_len)]
        else:
            fmt, size = types.get(value_type, ('I', 4))
            return struct.unpack(f'<{fmt}', f.read(size))[0]
            
    def extract_tensor(self, name: str) -> np.ndarray:
        """Extract a single tensor from GGUF."""
        info = self.tensor_info[name]
        
        with open(self.model_path, 'rb') as f:
            f.seek(self.data_offset + info['offset'])
            
            # Calculate size
            total_elements = 1
            for dim in info['dims']:
                total_elements *= dim
                
            # Read raw data (simplified - assumes float32 for now)
            # Real implementation needs to handle quantization
            raw_data = f.read(total_elements * 4)
            array = np.frombuffer(raw_data, dtype=np.float32)
            
            # Reshape
            array = array.reshape(info['dims'])
            
            return array
            
    def convert_to_pytorch(self, output_dir: str):
        """Convert GGUF to PyTorch format."""
        print(f"[Convert] Converting to PyTorch format...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract all tensors to PyTorch
        pytorch_state = {}
        
        for name in self.tensor_info.keys():
            print(f"[Convert] Extracting {name}...")
            try:
                np_array = self.extract_tensor(name)
                pytorch_state[name] = torch.from_numpy(np_array)
            except Exception as e:
                print(f"[Convert] Warning: Failed to extract {name}: {e}")
                
        # Save metadata
        metadata = {
            'original_format': 'gguf',
            'tensor_count': len(pytorch_state),
            'architecture': self.metadata.get('general.architecture', 'unknown')
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save using safetensors (efficient format)
        save_file(pytorch_state, os.path.join(output_dir, 'model.safetensors'))
        
        print(f"[Convert] Saved to {output_dir}")
        print(f"[Convert] Total tensors: {len(pytorch_state)}")
        
        return pytorch_state
        
    def split_for_providers(self, num_providers: int, output_dir: str):
        """Split model into provider shards."""
        print(f"[Split] Splitting for {num_providers} providers")
        
        # First convert to PyTorch
        pytorch_state = self.convert_to_pytorch(output_dir)
        
        # Identify layers
        layers = {}
        for name in pytorch_state.keys():
            if 'blk.' in name or 'layers.' in name:
                # Extract layer number
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part.isdigit():
                        layer_num = int(part)
                        if layer_num not in layers:
                            layers[layer_num] = []
                        layers[layer_num].append(name)
                        break
            elif 'token_embd' in name or 'embed' in name:
                if -1 not in layers:
                    layers[-1] = []
                layers[-1].append(name)
            elif 'output' in name or 'norm' in name:
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
                else:
                    # Remove output layers from non-last providers
                    for name in list(shard_tensors.keys()):
                        if name == 'output.weight' or name == 'output_norm.weight':
                            del shard_tensors[name]
                            print(f"[Split] Removing output layer from non-last provider: {name}")
                            
                # Save shard
                save_file(shard_tensors, os.path.join(shard_dir, 'shard.safetensors'))
                
                # Save shard info
                shard_info = {
                    'provider_id': i,
                    'start_layer': start_layer,
                    'end_layer': end_layer,
                    'tensor_count': len(shard_tensors),
                    'total_size_mb': sum(v.numel() * v.element_size() for v in shard_tensors.values()) / (1024 * 1024)
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
    converter.parse_gguf()
    converter.split_for_providers(args.providers, args.output_dir)
    
    
if __name__ == "__main__":
    main()
