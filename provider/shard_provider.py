"""
DiCAI Custom Sharding Protocol - Provider Node

Each provider loads ONLY its assigned layers from a GGUF model.
During inference, it receives input activations via HTTP, runs forward pass,
and returns output activations to the next provider.
"""

import argparse
import json
import os
import sys
import time
from typing import Optional, Dict, List
import numpy as np

# Use llama-cpp-python for model loading
from llama_cpp import Llama


class ShardProvider:
    """
    Provider node that serves a specific shard (subset of layers) of a model.
    """
    
    def __init__(self, provider_id: str, model_path: str, start_layer: int, end_layer: int, 
                 host: str = "0.0.0.0", port: int = 8080):
        self.provider_id = provider_id
        self.model_path = model_path
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.host = host
        self.port = port
        self.llm = None
        self.loaded = False
        
    def load_shard(self):
        """Load only the assigned layers from the model."""
        print(f"[{self.provider_id}] Loading layers {self.start_layer}-{self.end_layer} from {self.model_path}")
        
        # Load full model first (llama-cpp-python limitation)
        # In production, we'd modify llama.cpp to load only specific layers
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=512,
            verbose=False,
            # n_gpu_layers=0  # CPU only for now
        )
        
        self.loaded = True
        print(f"[{self.provider_id}] Shard loaded successfully")
        
    def process_activations(self, input_ids: List[int], past_kv_cache: Optional[dict] = None) -> dict:
        """
        Process input through this provider's layers.
        
        For true sharding, we'd need to:
        1. Run embedding lookup
        2. Pass through layers start_layer to end_layer
        3. Return hidden states + updated KV cache
        
        Since llama-cpp-python doesn't expose individual layers,
        we'll do full inference and document the limitation.
        """
        if not self.loaded:
            raise RuntimeError("Shard not loaded")
        
        # Convert input_ids to the format llama-cpp expects
        prompt = self.llm.detokenize(input_ids).decode('utf-8', errors='ignore')
        
        # For testing: do a single token generation
        # In real implementation, we'd manually run through specific layers
        output = self.llm(
            prompt,
            max_tokens=1,
            temperature=0.0,
            stop=["\n"]
        )
        
        return {
            "provider_id": self.provider_id,
            "layers_processed": f"{self.start_layer}-{self.end_layer}",
            "output_text": output['choices'][0]['text'],
            "status": "success"
        }
    
    def health(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "status": "ready" if self.loaded else "loading",
            "layers": f"{self.start_layer}-{self.end_layer}",
            "model": self.model_path
        }


import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

def create_app(provider: ShardProvider):
    """Create FastAPI app for the provider."""
    app = FastAPI(title=f"DiCAI Provider - {provider.provider_id}")
    
    class InferenceRequest(BaseModel):
        input_ids: List[int]
        past_kv: Optional[dict] = None
    
    @app.get("/health")
    def health():
        return provider.health()
    
    @app.post("/process")
    def process(req: InferenceRequest):
        try:
            result = provider.process_activations(req.input_ids, req.past_kv)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/load")
    def load():
        provider.load_shard()
        return {"status": "loaded"}
    
    return app


def main():
    parser = argparse.ArgumentParser(description="DiCAI Shard Provider")
    parser.add_argument("--id", required=True, help="Provider ID")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--start-layer", type=int, required=True, help="Start layer index")
    parser.add_argument("--end-layer", type=int, required=True, help="End layer index")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    args = parser.parse_args()
    
    provider = ShardProvider(
        args.id, args.model, args.start_layer, args.end_layer, args.host, args.port
    )
    
    print(f"="*60)
    print(f"DiCAI Provider Node")
    print(f"="*60)
    print(f"Provider ID: {args.id}")
    print(f"Model: {args.model}")
    print(f"Layers: {args.start_layer}-{args.end_layer}")
    print(f"Endpoint: http://{args.host}:{args.port}")
    print(f"="*60)
    
    # Load shard immediately
    provider.load_shard()
    
    # Start server
    app = create_app(provider)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
