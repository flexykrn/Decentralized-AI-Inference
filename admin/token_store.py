import json
import os
import time
from typing import Dict, Set

TOKENS_FILE = ".tokens.json"

class TokenStore:
    def __init__(self, filepath: str = TOKENS_FILE):
        self.filepath = filepath
        self.tokens: Dict[str, dict] = {}
        self._load()
    
    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                self.tokens = json.load(f)
    
    def _save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.tokens, f, indent=2)
    
    def create(self, token: str) -> dict:
        self.tokens[token] = {
            "created_at": time.time(),
            "status": "active"
        }
        self._save()
        return self.tokens[token]
    
    def validate(self, token: str) -> bool:
        return token in self.tokens and self.tokens[token].get("status") == "active"
    
    def revoke(self, token: str):
        if token in self.tokens:
            self.tokens[token]["status"] = "revoked"
            self._save()
    
    def list_active(self) -> list:
        return [t for t, info in self.tokens.items() if info.get("status") == "active"]
    
    def clear_all(self):
        self.tokens = {}
        self._save()
