#!/usr/bin/env python3
"""
DiCAI Distributed Inference - PyTorch Provider Node

Loads ONLY assigned layers and serves them via HTTP.
Receives activations from previous provider, runs forward pass,
returns output activations.
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
import torch.nn.functional as F
from safetensors.torch import load_file

# FastAPI for HTTP server
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


class DistributedLayer(nn.Module):
    """A single transformer layer that can be distributed."""
    
    def __init__(self, layer_weights: Dict[str, torch.Tensor], layer_num: int):
        super().__init__()
        self.layer_num = layer_num
        
        # Extract weights for this layer
        # Note: GGUF stores weights transposed, so we need to transpose them
        self.attention_q = self._get_weight(layer_weights, f'blk.{layer_num}.attn_q.weight')
        self.attention_k = self._get_weight(layer_weights, f'blk.{layer_num}.attn_k.weight')
        self.attention_v = self._get_weight(layer_weights, f'blk.{layer_num}.attn_v.weight')
        self.attention_o = self._get_weight(layer_weights, f'blk.{layer_num}.attn_output.weight')
        
        self.ffn_gate = self._get_weight(layer_weights, f'blk.{layer_num}.ffn_gate.weight')
        self.ffn_up = self._get_weight(layer_weights, f'blk.{layer_num}.ffn_up.weight')
        self.ffn_down = self._get_weight(layer_weights, f'blk.{layer_num}.ffn_down.weight')
        
        self.attention_norm = layer_weights.get(f'blk.{layer_num}.attn_norm.weight')
        self.ffn_norm = layer_weights.get(f'blk.{layer_num}.ffn_norm.weight')
        
    def _get_weight(self, weights: dict, name: str) -> Optional[torch.Tensor]:
        """Get weight from shard."""
        return weights.get(name)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through this layer."""
        residual = x
        batch_size, seq_len, hidden_dim = x.shape
        
        # Layer norm
        if self.attention_norm is not None:
            x = F.rms_norm(x, [hidden_dim], self.attention_norm, eps=1e-5)
            
        # Attention
        # Attention
        if self.attention_q is not None:
            # Q, K, V projections
            # For GGUF, weights are already in correct format [out_features, in_features]
            q = F.linear(x, self.attention_q)  # [batch, seq, hidden_dim]
            
            # For simplicity, use self-attention on Q only
            # In real implementation, we'd need proper multi-head attention with KV cache
            head_dim = q.shape[-1]
            scores = torch.matmul(q, q.transpose(-2, -1)) / (head_dim ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, q)
            
            # Project back to hidden dimension
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


class ShardProvider:
    """Provider node that serves specific layers."""
    
    def __init__(self, provider_id: str, shard_dir: str, start_layer: int, end_layer: int,
                 host: str = "0.0.0.0", port: int = 8080):
        self.provider_id = provider_id
        self.shard_dir = shard_dir
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.host = host
        self.port = port
        
        self.layers = nn.ModuleDict()
        self.embeddings = None
        self.output_norm = None
        self.output_projection = None
        self.loaded = False
        
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
                
        # Load output layers if present (only the final output layers, not layer-specific)
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
        
    def process(self, input_ids: Optional[List[int]] = None, hidden_states: Optional[List[List[float]]] = None) -> dict:
        """Process input through this provider's layers."""
        if not self.loaded:
            raise RuntimeError("Shard not loaded")
            
        # Convert to tensors
        if hidden_states is not None:
            x = torch.tensor(hidden_states, dtype=torch.float32)
            # Add batch dimension if missing
            if x.dim() == 2:
                x = x.unsqueeze(0)  # [1, seq_len, hidden_dim]
        else:
            # Use embeddings - input_ids select rows from embedding matrix
            if self.embeddings is None:
                raise RuntimeError("No embeddings available and no hidden states provided")
            # embeddings shape: [vocab_size, hidden_dim] (already correct from converter)
            x = self.embeddings[input_ids].unsqueeze(0)  # [1, seq_len, hidden_dim]
            
        print(f"[{self.provider_id}] Processing input shape: {x.shape}")
        
        # Run through layers
        for layer_num in range(self.start_layer, self.end_layer + 1):
            layer_key = str(layer_num)
            if layer_key in self.layers:
                x = self.layers[layer_key](x)
                print(f"[{self.provider_id}] Layer {layer_num} output shape: {x.shape}")
                
        # Apply output norm if present
        if self.output_norm is not None:
            print(f"[{self.provider_id}] Applying output norm with shape {self.output_norm.shape} to x shape {x.shape}")
            x = F.rms_norm(x, [x.shape[-1]], self.output_norm, eps=1e-5)
            
        # Project to vocabulary if present (only on last provider)
        logits = None
        if self.output_projection is not None:
            # output_projection shape: [hidden_dim, vocab_size]
            # x shape: [batch, seq, hidden_dim]
            logits = F.linear(x, self.output_projection)
            print(f"[{self.provider_id}] Logits shape: {logits.shape}")
            
        return {
            "provider_id": self.provider_id,
            "layers_processed": f"{self.start_layer}-{self.end_layer}",
            "hidden_states": x.squeeze(0).tolist(),
            "logits": logits.squeeze(0).tolist() if logits is not None else None,
            "status": "success"
        }
        
    def health(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "status": "ready" if self.loaded else "loading",
            "layers": f"{self.start_layer}-{self.end_layer}",
            "layer_count": len(self.layers)
        }


# FastAPI app
def create_app(provider: ShardProvider):
    app = FastAPI(title=f"DiCAI Provider - {provider.provider_id}")
    
    class ProcessRequest(BaseModel):
        input_ids: Optional[List[int]] = None
        hidden_states: Optional[List[List[float]]] = None
        
    @app.get("/health")
    def health():
        return provider.health()
        
    @app.post("/process")
    def process(req: ProcessRequest):
        try:
            result = provider.process(req.input_ids, req.hidden_states)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    @app.post("/load")
    def load():
        provider.load_shard()
        return {"status": "loaded"}
        
    return app


def main():
    parser = argparse.ArgumentParser(description="DiCAI PyTorch Provider")
    parser.add_argument("--id", required=True, help="Provider ID")
    parser.add_argument("--shard-dir", required=True, help="Directory containing shard")
    parser.add_argument("--start-layer", type=int, required=True, help="Start layer")
    parser.add_argument("--end-layer", type=int, required=True, help="End layer")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8080, help="Port")
    
    args = parser.parse_args()
    
    provider = ShardProvider(
        args.id, args.shard_dir, args.start_layer, args.end_layer, args.host, args.port
    )
    
    print(f"="*60)
    print(f"DiCAI PyTorch Provider")
    print(f"="*60)
    print(f"Provider ID: {args.id}")
    print(f"Shard: {args.shard_dir}")
    print(f"Layers: {args.start_layer}-{args.end_layer}")
    print(f"Endpoint: http://{args.host}:{args.port}")
    print(f"="*60)
    
    # Load shard
    provider.load_shard()
    
    # Start server
    app = create_app(provider)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
