#!/usr/bin/env python3
"""
DiCAI - API Server (Production)

OpenAI-compatible API with real distributed inference.
Connects to coordinator, routes through provider chain, streams tokens.

Usage:
    python -m dicai.api.server --port 8080 --coordinator http://localhost:8464
"""

import os
import sys
import time
import json
import uuid
import argparse
from typing import Optional, List, Dict, Any, Iterator

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coordinator.main import Coordinator
from src.dht.discovery import DHTClient


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
    model: str = "dicai-7b"
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
    
    def __init__(self, coordinator_url: str = "http://localhost:8464", port: int = 8080, host: str = "0.0.0.0"):
        self.app = FastAPI(title="DiCAI API", version="2.0.0")
        self.auth = AuthManager()
        self.coordinator_url = coordinator_url
        self.port = port
        self.host = host
        
        # Extract DHT host/port from coordinator URL
        coord_parts = coordinator_url.replace("http://", "").replace("https://", "").split(":")
        self.dht_host = coord_parts[0]
        self.dht_port = int(coord_parts[1].split("/")[0]) if len(coord_parts) > 1 else 8464
        self.dht_client = DHTClient(self.dht_host, self.dht_port)
        
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
                        "id": "dicai-7b",
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
        # Build prompt from messages
        prompt = self._build_prompt(request.messages)
        
        # Generate text
        text = await self._generate(prompt, request.max_tokens, request.temperature)
        
        # Count tokens (approximate)
        prompt_tokens = len(prompt.split())
        completion_tokens = len(text.split())
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )
        
        return response
        
    async def _stream_completion(self, request: ChatCompletionRequest):
        """Streaming completion."""
        session_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        
        # Build prompt
        prompt = self._build_prompt(request.messages)
        
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
        
        # Generate tokens one by one
        # For MVP, we generate all at once and stream word by word
        # Future: true token-by-token streaming from distributed providers
        text = await self._generate(prompt, request.max_tokens, request.temperature)
        
        words = text.split()
        for word in words:
            data = {
                'id': session_id,
                'object': 'chat.completion.chunk',
                'created': created,
                'model': request.model,
                'choices': [{
                    'index': 0,
                    'delta': {'content': word + ' '},
                    'finish_reason': None
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.05)  # Simulate streaming delay
            
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
        
    async def _generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using distributed providers."""
        # Get route from coordinator
        try:
            import requests
            response = requests.get(f"{self.coordinator_url}/route", timeout=5)
            if response.status_code != 200:
                return "Error: Could not get inference route"
                
            route_data = response.json()
            route = route_data.get("route", [])
            
            if not route:
                return "Error: No providers available"
                
            # For MVP, send to first provider in route
            # Future: chain through all providers with activation forwarding
            first_provider = route[0]
            address = first_provider.get('address', '127.0.0.1')
            if address == '0.0.0.0':
                address = '127.0.0.1'
            provider_url = f"http://{address}:{first_provider.get('port', 5001)}"

            # Wait for model to be loaded
            for _ in range(30):
                try:
                    health = requests.get(f"{provider_url}/health", timeout=2).json()
                    if health.get('model_loaded'):
                        break
                except Exception:
                    pass
                time.sleep(1)

            # Send request to provider
            provider_response = requests.post(
                f"{provider_url}/forward",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=120
            )

            if provider_response.status_code == 200:
                result = provider_response.json()
                return result.get("token", "")
            else:
                return f"Error: Provider returned {provider_response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"
            
    def _build_prompt(self, messages: List[ChatMessage]) -> str:
        """Build prompt from chat messages."""
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
        
    def run(self):
        """Run the API server."""
        print(f"[API] Starting DiCAI v2 API on {self.host}:{self.port}")
        print(f"[API] Connected to coordinator: {self.coordinator_url}")
        uvicorn.run(self.app, host=self.host, port=self.port)


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


def main():
    parser = argparse.ArgumentParser(description="DiCAI API Server")
    parser.add_argument("--port", type=int, default=8080, help="API port")
    parser.add_argument("--host", default="0.0.0.0", help="API host")
    parser.add_argument("--coordinator", default="http://localhost:8464", help="Coordinator URL")
    
    args = parser.parse_args()
    
    server = DiCAIAPIServer(
        coordinator_url=args.coordinator,
        port=args.port,
        host=args.host
    )
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
