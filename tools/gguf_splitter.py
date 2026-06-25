#!/usr/bin/env python3
"""
DiCAI GGUF Model Splitter - True Layer Extraction

Reads a GGUF file and extracts individual layer shards.
Each shard contains only the tensors for specific transformer layers.

Based on GGUF format specification:
https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


class GGUFSplitter:
    """
    Splits a GGUF model into layer-wise shards for distributed inference.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.metadata = {}
        self.tensors = {}
        self.tensor_info = {}
        
    def parse_gguf(self):
        """Parse GGUF file and extract metadata + tensor info."""
        print(f"[GGUF] Parsing {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError(f"Invalid GGUF magic: {magic}")
            
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            kv_count = struct.unpack('<Q', f.read(8))[0]
            
            print(f"[GGUF] Version: {version}, Tensors: {tensor_count}, KV pairs: {kv_count}")
            
            # Read metadata key-value pairs
            for _ in range(kv_count):
                key = self._read_string(f)
                value_type = struct.unpack('<I', f.read(4))[0]
                value = self._read_value(f, value_type)
                self.metadata[key] = value
                
            print(f"[GGUF] Metadata keys: {list(self.metadata.keys())[:10]}...")
            
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
                
            # Calculate data offset (where tensor data begins)
            self.data_offset = f.tell()
            
            print(f"[GGUF] Data offset: {self.data_offset}")
            print(f"[GGUF] Tensors: {list(self.tensor_info.keys())[:10]}...")
            
    def _read_string(self, f) -> str:
        """Read a GGUF string."""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')
        
    def _read_value(self, f, value_type: int):
        """Read a GGUF value based on type."""
        # GGUF value types
        types = {
            0: ('uint8', 1), 1: ('int8', 1), 2: ('uint16', 2), 3: ('int16', 2),
            4: ('uint32', 4), 5: ('int32', 4), 6: ('float32', 4), 7: ('bool', 1),
            8: ('string', None), 9: ('array', None), 10: ('uint64', 8),
            11: ('int64', 8), 12: ('float64', 8)
        }
        
        if value_type == 8:  # string
            return self._read_string(f)
        elif value_type == 9:  # array
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            return [self._read_value(f, arr_type) for _ in range(arr_len)]
        else:
            fmt, size = types.get(value_type, ('uint32', 4))
            if fmt in ('uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64'):
                # Map format to struct format character
                fmt_map = {
                    'uint8': 'B', 'int8': 'b',
                    'uint16': 'H', 'int16': 'h',
                    'uint32': 'I', 'int32': 'i',
                    'uint64': 'Q', 'int64': 'q'
                }
                struct_fmt = fmt_map.get(fmt, 'I')
                return struct.unpack(f'<{struct_fmt}', f.read(size))[0]
            elif fmt == 'float32':
                return struct.unpack('<f', f.read(size))[0]
            elif fmt == 'float64':
                return struct.unpack('<d', f.read(size))[0]
            elif fmt == 'bool':
                return struct.unpack('<?', f.read(size))[0]
                
    def identify_layers(self) -> Dict[int, List[str]]:
        """
        Identify which tensors belong to which layer.
        
        Typical llama structure:
        - token_embd.weight (layer 0 / embedding)
        - blk.0.attn_q.weight, blk.0.attn_k.weight, ... (layer 0)
        - blk.1.attn_q.weight, ... (layer 1)
        - output_norm.weight, output.weight (final layers)
        """
        layers = {}
        
        for name in self.tensor_info.keys():
            if name.startswith('blk.'):
                # Extract layer number: blk.0.attn_q.weight -> 0
                layer_num = int(name.split('.')[1])
                if layer_num not in layers:
                    layers[layer_num] = []
                layers[layer_num].append(name)
            elif 'token_embd' in name or 'position_embd' in name:
                # Embedding layer
                if -1 not in layers:
                    layers[-1] = []
                layers[-1].append(name)
            elif 'output' in name or 'output_norm' in name:
                # Output layer
                if 999 not in layers:
                    layers[999] = []
                layers[999].append(name)
                
        print(f"[GGUF] Identified {len(layers)} layer groups")
        for layer_num in sorted(layers.keys()):
            print(f"  Layer {layer_num}: {len(layers[layer_num])} tensors")
            
        return layers
        
    def extract_shard(self, layer_start: int, layer_end: int, output_path: str):
        """
        Extract a shard containing layers [layer_start, layer_end].
        
        This creates a new GGUF file with only the specified layers.
        """
        print(f"[GGUF] Extracting layers {layer_start}-{layer_end} to {output_path}")
        
        layers = self.identify_layers()
        
        # Collect tensors for this shard
        shard_tensors = {}
        
        # Always include embeddings (layer -1)
        if -1 in layers:
            for name in layers[-1]:
                shard_tensors[name] = self.tensor_info[name]
                
        # Include requested layers
        for layer_num in range(layer_start, layer_end + 1):
            if layer_num in layers:
                for name in layers[layer_num]:
                    shard_tensors[name] = self.tensor_info[name]
                    
        # Include output layer if this is the last shard
        if layer_end >= max([k for k in layers.keys() if k != 999], default=0):
            if 999 in layers:
                for name in layers[999]:
                    shard_tensors[name] = self.tensor_info[name]
                    
        print(f"[GGUF] Shard contains {len(shard_tensors)} tensors")
        
        # Write new GGUF file
        self._write_shard_gguf(output_path, shard_tensors)
        
    def _write_shard_gguf(self, output_path: str, shard_tensors: Dict):
        """Write a new GGUF file containing only the specified tensors."""
        
        with open(self.model_path, 'rb') as src:
            with open(output_path, 'wb') as dst:
                # Write header
                dst.write(b'GGUF')
                dst.write(struct.pack('<I', 3))  # version 3
                dst.write(struct.pack('<Q', len(shard_tensors)))  # tensor count
                
                # Filter metadata (keep architecture info, remove tokenizer)
                relevant_meta = {k: v for k, v in self.metadata.items() 
                               if not k.startswith('tokenizer') and not k.startswith('gguf.')
                               or k in ['general.architecture', 'general.name', 'general.quantization_version',
                                       'llama.context_length', 'llama.embedding_length', 'llama.block_count',
                                       'llama.feed_forward_length', 'llama.attention.head_count',
                                       'llama.attention.head_count_kv', 'llama.rope.dimension_count',
                                       'llama.attention.layer_norm_rms_epsilon']}
                dst.write(struct.pack('<Q', len(relevant_meta)))
                
                # Write metadata
                for key, value in relevant_meta.items():
                    self._write_string(dst, key)
                    self._write_value(dst, value)
                    
                # Calculate new data offset
                header_size = dst.tell()
                
                # Write tensor info
                new_offsets = {}
                current_offset = 0
                
                for name, info in sorted(shard_tensors.items()):
                    self._write_string(dst, name)
                    dst.write(struct.pack('<I', len(info['dims'])))
                    for dim in info['dims']:
                        dst.write(struct.pack('<Q', dim))
                    dst.write(struct.pack('<I', info['dtype']))
                    dst.write(struct.pack('<Q', current_offset))
                    
                    # Calculate tensor size (including alignment)
                    tensor_size = self._calculate_tensor_size(info)
                    # Align tensor size to 32 bytes
                    tensor_size = ((tensor_size + 31) // 32) * 32
                    new_offsets[name] = (info['offset'], tensor_size)
                    current_offset += tensor_size
                    
                # Align to 32 bytes before data
                padding = (32 - (dst.tell() % 32)) % 32
                dst.write(b'\x00' * padding)
                
                # Copy tensor data
                for name, info in sorted(shard_tensors.items()):
                    src_offset = self.data_offset + info['offset']
                    tensor_size = new_offsets[name][1]
                    
                    src.seek(src_offset)
                    data = src.read(tensor_size)
                    dst.write(data)
                    
        print(f"[GGUF] Shard written to {output_path}")
        print(f"[GGUF] Shard size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
    def _write_string(self, f, s: str):
        """Write a GGUF string."""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)
        
    def _write_value(self, f, value):
        """Write a GGUF value."""
        if isinstance(value, str):
            f.write(struct.pack('<I', 8))  # string type
            self._write_string(f, value)
        elif isinstance(value, int):
            f.write(struct.pack('<I', 5))  # int32 type
            f.write(struct.pack('<i', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', 6))  # float32 type
            f.write(struct.pack('<f', value))
        elif isinstance(value, list):
            f.write(struct.pack('<I', 9))  # array type
            if value and isinstance(value[0], str):
                f.write(struct.pack('<I', 8))  # string array
            elif value and isinstance(value[0], int):
                f.write(struct.pack('<I', 5))  # int32 array
            else:
                f.write(struct.pack('<I', 6))  # float32 array
            f.write(struct.pack('<Q', len(value)))
            for v in value:
                self._write_value(f, v)
                
    def _calculate_tensor_size(self, info: dict) -> int:
        """Calculate tensor size in bytes."""
        # GGUF dtype sizes (simplified)
        dtype_sizes = {
            0: 1,   # F32
            1: 1,   # F16
            2: 1,   # Q4_0
            3: 1,   # Q4_1
            6: 1,   # Q5_0
            7: 1,   # Q5_1
            8: 1,   # Q8_0
            9: 2,   # Q8_1
            10: 4,  # Q2_K
            11: 4,  # Q3_K
            12: 4,  # Q4_K
            13: 4,  # Q5_K
            14: 4,  # Q6_K
            15: 4,  # Q8_K
            16: 4,  # I8
            17: 4,  # I16
            18: 4,  # I32
            19: 4,  # I64
            20: 4,  # F64
            21: 4,  # IQ1_M
            22: 4,  # IQ1_S
            23: 4,  # IQ2_XXS
            24: 4,  # IQ2_XS
            25: 4,  # IQ2_S
            26: 4,  # IQ2_M
            27: 4,  # IQ3_XXS
            28: 4,  # IQ3_XS
            29: 4,  # IQ4_XS
            30: 4,  # IQ4_NL
            31: 4,  # IQ3_S
            32: 4,  # IQ3_M
            33: 4,  # IQ4_K
            34: 4,  # IQ5_K
            35: 4,  # IQ6_K
            36: 4,  # IQ4_K_S
        }
        
        # Calculate total elements
        total_elements = 1
        for dim in info['dims']:
            total_elements *= dim
            
        # For quantized types, we need to handle block sizes
        # This is a simplified calculation - real GGUF uses block quantization
        dtype = info['dtype']
        if dtype in dtype_sizes:
            return total_elements * dtype_sizes[dtype]
        else:
            return total_elements * 4  # default to 4 bytes
            
    def split_for_providers(self, num_providers: int, output_dir: str):
        """
        Split model into shards for N providers.
        
        Each provider gets approximately equal number of layers.
        """
        print(f"[GGUF] Splitting for {num_providers} providers")
        
        layers = self.identify_layers()
        
        # Get actual layer numbers (excluding -1 and 999)
        layer_nums = sorted([k for k in layers.keys() if k >= 0 and k < 999])
        
        if not layer_nums:
            print("[GGUF] ERROR: No layers found!")
            return
            
        total_layers = len(layer_nums)
        layers_per_provider = total_layers // num_providers
        remainder = total_layers % num_providers
        
        print(f"[GGUF] Total layers: {total_layers}")
        print(f"[GGUF] Layers per provider: {layers_per_provider}, remainder: {remainder}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        start_idx = 0
        for i in range(num_providers):
            # Distribute remainder across first few providers
            extra = 1 if i < remainder else 0
            end_idx = start_idx + layers_per_provider + extra - 1
            
            if start_idx < len(layer_nums):
                start_layer = layer_nums[start_idx]
                end_layer = layer_nums[min(end_idx, len(layer_nums) - 1)]
                
                output_path = os.path.join(output_dir, f"shard_{i}_{start_layer}_{end_layer}.gguf")
                self.extract_shard(start_layer, end_layer, output_path)
                
                start_idx = end_idx + 1
                
        print(f"[GGUF] All shards written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DiCAI GGUF Model Splitter")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--output-dir", default="shards", help="Output directory for shards")
    parser.add_argument("--providers", type=int, default=2, help="Number of providers")
    parser.add_argument("--layer-start", type=int, help="Start layer (for single shard)")
    parser.add_argument("--layer-end", type=int, help="End layer (for single shard)")
    parser.add_argument("--output", help="Output path (for single shard)")
    
    args = parser.parse_args()
    
    splitter = GGUFSplitter(args.model)
    splitter.parse_gguf()
    
    if args.layer_start is not None and args.layer_end is not None and args.output:
        # Extract single shard
        splitter.extract_shard(args.layer_start, args.layer_end, args.output)
    else:
        # Split for multiple providers
        splitter.split_for_providers(args.providers, args.output_dir)


if __name__ == "__main__":
    main()
