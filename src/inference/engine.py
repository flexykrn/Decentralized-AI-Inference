#!/usr/bin/env python3
"""
DiCAI - Inference Engine

Real inference using llama.cpp for model loading and token generation.
"""

import os
import sys
import time
from typing import List, Optional, Iterator

import torch
import numpy as np

# Try to import llama_cpp
try:
    import llama_cpp
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    print("[WARNING] llama.cpp not installed. Install with: pip install llama-cpp-python")

from src.memory.loader import OptimizedShardLoader


class InferenceEngine:
    """Real inference engine for token generation."""
    
    def __init__(self, model_path: str, shard_dir: Optional[str] = None):
        self.model_path = model_path
        self.shard_dir = shard_dir
        self.model = None
        self.tokenizer = None
        
        # Try llama.cpp first
        if HAS_LLAMA_CPP and model_path.endswith('.gguf'):
            self._load_llama_cpp()
        elif shard_dir:
            # Use PyTorch shards
            self._load_pytorch()
        else:
            raise ValueError("No model or shards provided")
            
    def _load_llama_cpp(self):
        """Load model using llama.cpp."""
        print(f"[Engine] Loading model with llama.cpp: {self.model_path}")
        
        self.model = llama_cpp.Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_batch=512,
            verbose=False
        )
        
        print(f"[Engine] Model loaded: {self.model.model_params.n_vocab} vocab")
        
    def _load_pytorch(self):
        """Load PyTorch shards."""
        print(f"[Engine] Loading PyTorch shards from: {self.shard_dir}")
        
        self.loader = OptimizedShardLoader(self.shard_dir)
        # TODO: Implement PyTorch inference
        raise NotImplementedError("PyTorch inference not yet implemented")
        
    def generate(self, prompt: str, max_tokens: int = 100, 
                 temperature: float = 0.7, stream: bool = False) -> Iterator[str]:
        """Generate tokens from prompt."""
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        if HAS_LLAMA_CPP and hasattr(self, 'model'):
            # Use llama.cpp
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>"],
                stream=stream
            )
            
            if stream:
                for chunk in output:
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        text = chunk['choices'][0].get('text', '')
                        if text:
                            yield text
            else:
                text = output['choices'][0]['text']
                yield text
        else:
            raise RuntimeError("No inference backend available")
            
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to IDs."""
        if self.model:
            return self.model.tokenize(text.encode())
        return []
        
    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize IDs to text."""
        if self.model:
            return self.model.detokenize(tokens).decode('utf-8', errors='ignore')
        return ""


def test_engine():
    """Test inference engine."""
    print("=" * 60)
    print("Inference Engine Test")
    print("=" * 60)
    
    # Check for model
    model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Skipping test - no model available")
        return True
        
    if not HAS_LLAMA_CPP:
        print("llama.cpp not installed")
        print("Install with: pip install llama-cpp-python")
        return True
        
    # Load engine
    engine = InferenceEngine(model_path)
    
    # Test tokenization
    tokens = engine.tokenize("Hello world")
    print(f"Tokenized: {tokens}")
    
    # Test generation
    print("\nGenerating...")
    start = time.time()
    
    response = engine.generate("What is the capital of France?", max_tokens=20)
    text = ''.join(response)
    
    elapsed = time.time() - start
    print(f"Generated: {text}")
    print(f"Time: {elapsed:.2f}s")
    
    print("\nInference Engine Test PASSED")
    return True


if __name__ == "__main__":
    test_engine()
