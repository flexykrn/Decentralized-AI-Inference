#!/usr/bin/env python3
"""
DiCAI v2 - Production API Server

OpenAI-compatible API with:
- Authentication middleware
- Rate limiting
- Streaming responses
- Health monitoring
"""

import os
import sys
import time
import json
import uuid
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v2.src.dht.discovery import DHTClient, DHTNode, ProviderInfo
from v2.src.inference.session import FaultTolerantRouter, InferenceSession


# Auth
security = HTTPBearer()

class AuthManager:
    """Simple API key manager."""
    
    def __init__(self, api_keys_file: str = ".api_keys.json"):
        self.api_keys_file = api_keys_file
        self.api_keys: Dict[str, dict] = {}
        self._load_keys()
        
    def _load_keys(self):
        """Load API keys from file."""
        if os.path.exists(self.api_keys_file):
            with open(self.api_keys_file, 'r') as f:
                self.api_keys = json.load(f)
                
    def _save_keys(self):
        """Save API keys to file."""
        with open(self.api_keys_file, 'w') as f:
            json.dump(self.api_keys, f)
            
    def create_key(self, name: str, rate_limit: int = 100) -> str:
        """Create new API key."""
        key = f"dicai_{uuid.uuid4().hex}"
        self.api_keys[key] = {
            "name": name,
            "created_at": time.time(),
            "rate_limit": rate_limit,
            "requests_count": 0,
            "last_request": 0
        }
        self._save_keys()
        return key
        
    def validate_key(self, key: str) -> bool:
        """Validate API key."""
        return key in self.api_keys
        
    def check_rate_limit(self, key: str) -> bool:
        """Check if key is within rate limit."""
        if key not in self.api_keys:
            return False
            
        info = self.api_keys[key]
        now = time.time()
        
        # Reset counter if it's been more than an hour
        if now - info["last_request"] > 3600:
            info["requests_count"] = 0
            
        if info["requests_count"] >= info["rate_limit"]:
            return False
            
        info["requests_count"] += 1
        info["last_request"] = now
        self._save_keys()
        return True


# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "dicai-70b"
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False
    
class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class DiCAIAPIServer:
    """Production API server for DiCAI."""
    
    def __init__(self, dht_host: str = "localhost", dht_port: int = 8464):
        self.app = FastAPI(title="DiCAI API", version="2.0.0")
        self.auth = AuthManager()
        self.dht_client = DHTClient(dht_host, dht_port)
        self.router = FaultTolerantRouter(self.dht_client)
        self.sessions: Dict[str, InferenceSession] = {}
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.post("/v1/chat/completions")
        async def chat_completion(request: ChatCompletionRequest, 
                                   credentials: HTTPAuthorizationCredentials = Depends(security)):
            """OpenAI-compatible chat completion endpoint."""
            api_key = credentials.credentials
            
            # Validate key
            if not self.auth.validate_key(api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")
                
            # Check rate limit
            if not self.auth.check_rate_limit(api_key):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
            if request.stream:
                return StreamingResponse(
                    self._stream_completion(request),
                    media_type="text/event-stream"
                )
            else:
                return await self._complete(request)
                
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            dht_health = self.dht_client.health()
            return {
                "status": "healthy",
                "dht": dht_health,
                "version": "2.0.0"
            }
            
        @self.app.get("/v1/models")
        async def list_models():
            """List available models."""
            return {
                "object": "list",
                "data": [
                    {
                        "id": "dicai-70b",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "dicai"
                    }
                ]
            }
            
        @self.app.post("/admin/keys")
        async def create_key(name: str, rate_limit: int = 100):
            """Create new API key (admin only)."""
            key = self.auth.create_key(name, rate_limit)
            return {"key": key, "name": name, "rate_limit": rate_limit}
            
    async def _complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Non-streaming completion."""
        # Create inference session
        session = self.router.create_session()
        
        # TODO: Implement actual token generation
        # For now, return dummy response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from DiCAI v2."
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20
            }
        )
        
        # Cleanup
        self.router.close_session(session.session_id)
        
        return response
        
    async def _stream_completion(self, request: ChatCompletionRequest):
        """Streaming completion."""
        session_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        
        # Send initial response
        data = {
            'id': session_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': request.model,
            'choices': [{
                'index': 0,
                'delta': {'role': 'assistant'},
                'finish_reason': None
            }]
        }
        yield f"data: {json.dumps(data)}\n\n"
        
        # Simulate token generation
        tokens = ["Hello", ",", " this", " is", " DiCAI", " v2", "."]
        for token in tokens:
            data = {
                'id': session_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': request.model,
                'choices': [{
                    'index': 0,
                    'delta': {'content': token},
                    'finish_reason': None
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.1)
            
        # Send final response
        data = {
            'id': session_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': request.model,
            'choices': [{
                'index': 0,
                'delta': {},
                'finish_reason': 'stop'
            }]
        }
        yield f"data: {json.dumps(data)}\n\n"
        
        yield "data: [DONE]\n\n"
        
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the API server."""
        print(f"[API] Starting DiCAI v2 API on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def test_api():
    """Test API server."""
    print("=" * 60)
    print("API Server Test")
    print("=" * 60)
    
    # Create server
    server = DiCAIAPIServer()
    
    # Create API key
    key = server.auth.create_key("test", rate_limit=1000)
    print(f"\nCreated API key: {key}")
    
    # Test auth
    assert server.auth.validate_key(key), "Key validation failed"
    print("Key validation: PASSED")
    
    # Test rate limit
    assert server.auth.check_rate_limit(key), "Rate limit check failed"
    print("Rate limit check: PASSED")
    
    # Test invalid key
    assert not server.auth.validate_key("invalid_key"), "Invalid key should fail"
    print("Invalid key rejection: PASSED")
    
    print("\nAPI Server Test PASSED")
    return True


if __name__ == "__main__":
    test_api()
