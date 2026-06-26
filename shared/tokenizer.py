#!/usr/bin/env python3
"""
DiCAI Tokenizer

Handles text -> tokens and tokens -> text conversion.
Supports SentencePiece (Llama) and TikToken (GPT).
"""

import os
import json
from typing import List, Optional


class Tokenizer:
    """Generic tokenizer wrapper."""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tokenizer_type = None
        self.vocab_size = 32000
        self.bos_token = 1
        self.eos_token = 2
        self.pad_token = 0
        
        # Detect tokenizer type
        self._detect_tokenizer()
        
    def _detect_tokenizer(self):
        """Auto-detect tokenizer type from model directory."""
        # Check for SentencePiece model
        sp_model = os.path.join(self.model_dir, "tokenizer.model")
        if os.path.exists(sp_model):
            self.tokenizer_type = "sentencepiece"
            self._init_sentencepiece(sp_model)
            return
            
        # Check for TikToken
        tiktoken_file = os.path.join(self.model_dir, "tokenizer.json")
        if os.path.exists(tiktoken_file):
            self.tokenizer_type = "tiktoken"
            self._init_tiktoken(tiktoken_file)
            return
            
        # Default: simple character-based (for testing)
        self.tokenizer_type = "simple"
        print(f"[Tokenizer] No tokenizer found, using simple char-based")
        
    def _init_sentencepiece(self, model_path: str):
        """Initialize SentencePiece tokenizer."""
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model_path)
            self.vocab_size = self.sp.vocab_size()
            self.bos_token = self.sp.bos_id()
            self.eos_token = self.sp.eos_id()
            self.pad_token = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0
            print(f"[Tokenizer] Loaded SentencePiece: vocab_size={self.vocab_size}")
        except Exception as e:
            print(f"[Tokenizer] Failed to load SentencePiece: {e}")
            self.tokenizer_type = "simple"
            
    def _init_tiktoken(self, json_path: str):
        """Initialize TikToken tokenizer."""
        try:
            import tiktoken
            with open(json_path) as f:
                data = json.load(f)
            self.encoding = tiktoken.get_encoding(data.get("model_name", "cl100k_base"))
            self.vocab_size = self.encoding.n_vocab
            print(f"[Tokenizer] Loaded TikToken: vocab_size={self.vocab_size}")
        except Exception as e:
            print(f"[Tokenizer] Failed to load TikToken: {e}")
            self.tokenizer_type = "simple"
            
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs."""
        if self.tokenizer_type == "sentencepiece":
            tokens = self.sp.Encode(text, add_bos=add_bos, add_eos=add_eos)
            return tokens
        elif self.tokenizer_type == "tiktoken":
            tokens = self.encoding.encode(text)
            if add_bos:
                tokens = [self.bos_token] + tokens
            if add_eos:
                tokens = tokens + [self.eos_token]
            return tokens
        else:
            # Simple character-based fallback
            tokens = [ord(c) % self.vocab_size for c in text]
            if add_bos:
                tokens = [self.bos_token] + tokens
            return tokens
            
    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        if self.tokenizer_type == "sentencepiece":
            if skip_special:
                tokens = [t for t in tokens if t not in [self.bos_token, self.eos_token, self.pad_token]]
            return self.sp.Decode(tokens)
        elif self.tokenizer_type == "tiktoken":
            if skip_special:
                tokens = [t for t in tokens if t not in [self.bos_token, self.eos_token, self.pad_token]]
            return self.encoding.decode(tokens)
        else:
            # Simple character-based fallback
            if skip_special:
                tokens = [t for t in tokens if t not in [self.bos_token, self.eos_token, self.pad_token]]
            return "".join([chr(t % 128) for t in tokens if t < 128])
            
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
        
    def get_special_tokens(self) -> dict:
        """Get special token IDs."""
        return {
            "bos": self.bos_token,
            "eos": self.eos_token,
            "pad": self.pad_token
        }


def test_tokenizer():
    """Test tokenizer functionality."""
    # Create a simple test
    tokenizer = Tokenizer(".")  # Will use simple fallback
    
    text = "Hello world"
    tokens = tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
    
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: {tokenizer.get_special_tokens()}")


if __name__ == "__main__":
    test_tokenizer()
