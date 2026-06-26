#!/usr/bin/env python3
"""
DiCAI Authentication Manager

Handles invite tokens, provider registration, and API key management.
"""

import secrets
import hashlib
import time
import json
import os
from typing import Optional, Dict, List


class AuthManager:
    """Manages authentication for providers and clients."""
    
    def __init__(self, token_file: str = ".tokens.json"):
        self.token_file = token_file
        self.invite_tokens: Dict[str, dict] = {}  # token -> {created_at, expires_at, used, used_by}
        self.provider_tokens: Dict[str, str] = {}  # provider_id -> token
        self.api_keys: Dict[str, dict] = {}  # api_key -> {created_at, rate_limit}
        self.load_tokens()
        
    def load_tokens(self):
        """Load existing tokens from file."""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file) as f:
                    data = json.load(f)
                    self.invite_tokens = data.get("invite_tokens", {})
                    self.provider_tokens = data.get("provider_tokens", {})
                    self.api_keys = data.get("api_keys", {})
            except Exception as e:
                print(f"[Auth] Failed to load tokens: {e}")
                
    def save_tokens(self):
        """Save tokens to file."""
        data = {
            "invite_tokens": self.invite_tokens,
            "provider_tokens": self.provider_tokens,
            "api_keys": self.api_keys
        }
        with open(self.token_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def generate_invite_token(self, expires_hours: int = 24) -> str:
        """Generate a new invite token for provider registration."""
        token = secrets.token_urlsafe(32)
        self.invite_tokens[token] = {
            "created_at": time.time(),
            "expires_at": time.time() + expires_hours * 3600,
            "used": False,
            "used_by": None
        }
        self.save_tokens()
        print(f"[Auth] Generated invite token (expires in {expires_hours}h)")
        return token
        
    def validate_invite_token(self, token: str) -> bool:
        """Validate an invite token."""
        if token not in self.invite_tokens:
            print(f"[Auth] Invalid token: not found")
            return False
            
        token_data = self.invite_tokens[token]
        
        if token_data["used"]:
            print(f"[Auth] Token already used")
            return False
            
        if time.time() > token_data["expires_at"]:
            print(f"[Auth] Token expired")
            return False
            
        return True
        
    def use_invite_token(self, token: str, provider_id: str) -> bool:
        """Mark an invite token as used."""
        if not self.validate_invite_token(token):
            return False
            
        self.invite_tokens[token]["used"] = True
        self.invite_tokens[token]["used_by"] = provider_id
        self.provider_tokens[provider_id] = token
        self.save_tokens()
        print(f"[Auth] Token used by provider {provider_id}")
        return True
        
    def generate_api_key(self, rate_limit: int = 100) -> str:
        """Generate API key for client access."""
        api_key = f"dicai_{secrets.token_urlsafe(32)}"
        self.api_keys[api_key] = {
            "created_at": time.time(),
            "rate_limit": rate_limit,
            "requests": 0
        }
        self.save_tokens()
        print(f"[Auth] Generated API key (rate limit: {rate_limit}/min)")
        return api_key
        
    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key."""
        if api_key not in self.api_keys:
            return False
            
        key_data = self.api_keys[api_key]
        
        # Check rate limit (simple: reset every minute)
        if time.time() - key_data["created_at"] > 60:
            key_data["requests"] = 0
            key_data["created_at"] = time.time()
            
        if key_data["requests"] >= key_data["rate_limit"]:
            print(f"[Auth] Rate limit exceeded for API key")
            return False
            
        key_data["requests"] += 1
        self.save_tokens()
        return True
        
    def list_tokens(self) -> dict:
        """List all tokens and their status."""
        return {
            "invite_tokens": len(self.invite_tokens),
            "active_invites": sum(1 for t in self.invite_tokens.values() if not t["used"]),
            "providers": len(self.provider_tokens),
            "api_keys": len(self.api_keys)
        }
        
    def revoke_provider(self, provider_id: str):
        """Revoke a provider's access."""
        if provider_id in self.provider_tokens:
            del self.provider_tokens[provider_id]
            self.save_tokens()
            print(f"[Auth] Revoked provider {provider_id}")
            
    def revoke_api_key(self, api_key: str):
        """Revoke an API key."""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            self.save_tokens()
            print(f"[Auth] Revoked API key")


def test_auth():
    """Test authentication manager."""
    auth = AuthManager(".test_tokens.json")
    
    # Generate invite token
    invite = auth.generate_invite_token(expires_hours=1)
    print(f"Invite token: {invite[:20]}...")
    
    # Validate
    assert auth.validate_invite_token(invite) == True
    print("Token validation: PASSED")
    
    # Use token
    assert auth.use_invite_token(invite, "p1") == True
    print("Token usage: PASSED")
    
    # Try to use again
    assert auth.validate_invite_token(invite) == False
    print("Token reuse blocked: PASSED")
    
    # Generate API key
    api_key = auth.generate_api_key(rate_limit=10)
    print(f"API key: {api_key[:20]}...")
    
    # Validate API key
    assert auth.validate_api_key(api_key) == True
    print("API key validation: PASSED")
    
    # List tokens
    stats = auth.list_tokens()
    print(f"Token stats: {stats}")
    
    # Cleanup
    os.remove(".test_tokens.json")
    print("Auth test PASSED")


if __name__ == "__main__":
    test_auth()
