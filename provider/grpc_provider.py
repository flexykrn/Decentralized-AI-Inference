#!/usr/bin/env python3
"""
DiCAI gRPC Provider Node

Loads ONLY assigned layers and serves them via gRPC.
Receives activations from coordinator, runs forward pass,
returns hidden states to coordinator for next provider.
"""

import argparse
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

import grpc
from concurrent import futures

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import proto.dicai_pb2 as dicai_pb2
import proto.dicai_pb2_grpc as dicai_pb2_grpc


class DistributedLayer(nn.Module):
    """A single transformer layer that can be distributed."""
    
    def __init__(self, layer_weights, layer_num):
        super().__init__()
        self.layer_num = layer_num
        
        # Extract weights for this layer
        self.attention_q = layer_weights.get(f'blk.{layer_num}.attn_q.weight')
        self.attention_k = layer_weights.get(f'blk.{layer_num}.attn_k.weight')
        self.attention_v = layer_weights.get(f'blk.{layer_num}.attn_v.weight')
        self.attention_o = layer_weights.get(f'blk.{layer_num}.attn_output.weight')
        
        self.ffn_gate = layer_weights.get(f'blk.{layer_num}.ffn_gate.weight')
        self.ffn_up = layer_weights.get(f'blk.{layer_num}.ffn_up.weight')
        self.ffn_down = layer_weights.get(f'blk.{layer_num}.ffn_down.weight')
        
        self.attention_norm = layer_weights.get(f'blk.{layer_num}.attn_norm.weight')
        self.ffn_norm = layer_weights.get(f'blk.{layer_num}.ffn_norm.weight')
        
    def forward(self, x, mask=None):
        """Forward pass through this layer."""
        residual = x
        batch_size, seq_len, hidden_dim = x.shape
        
        # Layer norm
        if self.attention_norm is not None:
            x = F.rms_norm(x, [hidden_dim], self.attention_norm, eps=1e-5)
            
        # Attention
        if self.attention_q is not None:
            q = F.linear(x, self.attention_q)
            
            # Self-attention on Q (simplified)
            head_dim = q.shape[-1]
            scores = torch.matmul(q, q.transpose(-2, -1)) / (head_dim ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, q)
            
            # Project back
            if self.attention_o is not None:
                attn_output = F.linear(attn_output, self.attention_o)
                
            x = residual + attn_output
            
        # FFN
        residual = x
        if self.ffn_norm is not None:
            x = F.rms_norm(x, [hidden_dim], self.ffn_norm, eps=1e-5)
            
        if self.ffn_gate is not None and self.ffn_up is not None:
            gate = F.silu(F.linear(x, self.ffn_gate))
            up = F.linear(x, self.ffn_up)
            x = gate * up
            
        if self.ffn_down is not None:
            x = F.linear(x, self.ffn_down)
            
        x = residual + x
        
        return x


class ProviderServicer(dicai_pb2_grpc.ProviderServiceServicer):
    """gRPC servicer for provider nodes."""
    
    def __init__(self, provider_id, shard_dir, start_layer, end_layer):
        self.provider_id = provider_id
        self.shard_dir = shard_dir
        self.start_layer = start_layer
        self.end_layer = end_layer
        
        self.layers = nn.ModuleDict()
        self.embeddings = None
        self.output_norm = None
        self.output_projection = None
        self.loaded = False
        
        self.load_shard()
        
    def load_shard(self):
        """Load the shard from disk."""
        print(f"[{self.provider_id}] Loading shard from {self.shard_dir}")
        
        shard_path = os.path.join(self.shard_dir, 'shard.safetensors')
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard not found: {shard_path}")
            
        weights = load_file(shard_path)
        
        print(f"[{self.provider_id}] Loaded {len(weights)} tensors")
        
        # Load embeddings if present
        for name, tensor in weights.items():
            if 'token_embd' in name or 'embed' in name:
                self.embeddings = tensor
                print(f"[{self.provider_id}] Loaded embeddings: {tensor.shape}")
                
        # Load output layers if present
        for name, tensor in weights.items():
            if name == 'output_norm.weight':
                self.output_norm = tensor
                print(f"[{self.provider_id}] Loaded output norm: {tensor.shape}")
            elif name == 'output.weight':
                self.output_projection = tensor
                print(f"[{self.provider_id}] Loaded output projection: {tensor.shape}")
                
        # Load transformer layers
        for layer_num in range(self.start_layer, self.end_layer + 1):
            layer_weights = {}
            for name, tensor in weights.items():
                if f'blk.{layer_num}.' in name or f'layers.{layer_num}.' in name:
                    layer_weights[name] = tensor
                    
            if layer_weights:
                self.layers[str(layer_num)] = DistributedLayer(layer_weights, layer_num)
                print(f"[{self.provider_id}] Loaded layer {layer_num}")
                
        self.loaded = True
        print(f"[{self.provider_id}] Shard loaded successfully")
        
    def process(self, request, context):
        """Process input through this provider's layers."""
        if not self.loaded:
            return dicai_pb2.ProcessResponse(
                status="error",
                error="Shard not loaded"
            )
            
        try:
            # Convert to tensors
            if request.hidden_states:
                # Deserialize hidden states
                hidden_np = np.frombuffer(request.hidden_states, dtype=np.float32)
                hidden_shape = list(request.hidden_states_shape)
                x = torch.from_numpy(hidden_np.reshape(hidden_shape))
            else:
                # Use embeddings
                if self.embeddings is None:
                    return dicai_pb2.ProcessResponse(
                        status="error",
                        error="No embeddings available and no hidden states provided"
                    )
                input_ids = list(request.input_ids)
                x = self.embeddings[input_ids].unsqueeze(0)
                
            print(f"[{self.provider_id}] Processing input shape: {x.shape}")
            
            # Run through layers
            for layer_num in range(self.start_layer, self.end_layer + 1):
                layer_key = str(layer_num)
                if layer_key in self.layers:
                    x = self.layers[layer_key](x)
                    
            # Apply output norm if present
            if self.output_norm is not None:
                x = F.rms_norm(x, [x.shape[-1]], self.output_norm, eps=1e-5)
                
            # Project to vocabulary if present
            logits = None
            logits_shape = None
            if self.output_projection is not None:
                logits = F.linear(x, self.output_projection)
                logits_shape = list(logits.shape)
                logits = logits.detach().cpu().numpy().tobytes()
                
            # Serialize hidden states
            hidden_states = x.detach().cpu().numpy()
            hidden_shape = list(hidden_states.shape)
            hidden_bytes = hidden_states.tobytes()
            
            return dicai_pb2.ProcessResponse(
                status="success",
                provider_id=self.provider_id,
                layers_processed=f"{self.start_layer}-{self.end_layer}",
                hidden_states=hidden_bytes,
                hidden_states_shape=hidden_shape,
                logits=logits if logits is not None else b"",
                logits_shape=logits_shape if logits_shape else [],
                request_id=request.request_id
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return dicai_pb2.ProcessResponse(
                status="error",
                error=str(e),
                request_id=request.request_id
            )
            
    def health(self, request, context):
        """Health check."""
        import psutil
        mem = psutil.virtual_memory()
        
        return dicai_pb2.HealthResponse(
            provider_id=self.provider_id,
            status="ready" if self.loaded else "loading",
            layers=f"{self.start_layer}-{self.end_layer}",
            layer_count=len(self.layers),
            memory_used_mb=mem.used // (1024 * 1024),
            memory_total_mb=mem.total // (1024 * 1024),
            gpu_info="cpu"
        )
        
    def processStream(self, request_iterator, context):
        """Stream processing."""
        for request in request_iterator:
            yield self.process(request, context)


def serve(provider_id, shard_dir, start_layer, end_layer, port):
    """Start gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    servicer = ProviderServicer(provider_id, shard_dir, start_layer, end_layer)
    dicai_pb2_grpc.add_ProviderServiceServicer_to_server(servicer, server)
    
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    print(f"\n{'='*60}")
    print(f"DiCAI gRPC Provider")
    print(f"{'='*60}")
    print(f"Provider ID: {provider_id}")
    print(f"Shard: {shard_dir}")
    print(f"Layers: {start_layer}-{end_layer}")
    print(f"gRPC Port: {port}")
    print(f"{'='*60}\n")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print(f"\n[{provider_id}] Shutting down...")
        server.stop(0)


def main():
    parser = argparse.ArgumentParser(description="DiCAI gRPC Provider")
    parser.add_argument("--id", required=True, help="Provider ID")
    parser.add_argument("--shard-dir", required=True, help="Shard directory")
    parser.add_argument("--start-layer", type=int, required=True, help="Start layer")
    parser.add_argument("--end-layer", type=int, required=True, help="End layer")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port")
    
    args = parser.parse_args()
    
    serve(args.id, args.shard_dir, args.start_layer, args.end_layer, args.port)


if __name__ == "__main__":
    main()
