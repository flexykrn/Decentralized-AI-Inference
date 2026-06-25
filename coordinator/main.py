#!/usr/bin/env python3
"""
Distributed Inference Coordinator
Routes requests across multiple llama.cpp provider instances.
"""
import asyncio
import json
import random
import sys
import time
from typing import Dict, List, Optional

import aiohttp
from aiohttp import web

# Provider registry: {provider_id: {host, port, status, last_seen}}
providers: Dict[str, dict] = {}

# Round-robin index
rr_index = 0


async def register_provider(request):
    """Register a new provider."""
    data = await request.json()
    provider_id = data.get("id", f"p{len(providers)+1}")
    
    providers[provider_id] = {
        "id": provider_id,
        "host": data.get("host", "localhost"),
        "port": data.get("port", 8081),
        "status": "online",
        "last_seen": time.time(),
    }
    
    print(f"[Coordinator] Provider {provider_id} registered: {data.get('host')}:{data.get('port')}")
    return web.json_response({"status": "registered", "id": provider_id})


async def list_providers(request):
    """List all registered providers."""
    return web.json_response({
        "providers": list(providers.values()),
        "count": len(providers),
    })


async def health(request):
    """Health check."""
    return web.json_response({
        "status": "ok",
        "providers": len(providers),
    })


async def get_next_provider() -> Optional[dict]:
    """Get next available provider using round-robin."""
    global rr_index
    
    online = [p for p in providers.values() if p["status"] == "online"]
    if not online:
        return None
    
    provider = online[rr_index % len(online)]
    rr_index += 1
    return provider


async def proxy_chat(request):
    """Proxy chat completion request to a provider."""
    provider = await get_next_provider()
    if not provider:
        return web.json_response(
            {"error": "No providers available"},
            status=503
        )
    
    # Read the request body
    body = await request.read()
    
    # Forward to provider
    target_url = f"http://{provider['host']}:{provider['port']}/v1/chat/completions"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                target_url,
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                response_body = await resp.read()
                return web.Response(
                    body=response_body,
                    status=resp.status,
                    content_type="application/json"
                )
        except Exception as e:
            provider["status"] = "error"
            return web.json_response(
                {"error": f"Provider error: {str(e)}"},
                status=502
            )


async def proxy_completions(request):
    """Proxy completion request to a provider."""
    provider = await get_next_provider()
    if not provider:
        return web.json_response(
            {"error": "No providers available"},
            status=503
        )
    
    body = await request.read()
    target_url = f"http://{provider['host']}:{provider['port']}/v1/completions"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                target_url,
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                response_body = await resp.read()
                return web.Response(
                    body=response_body,
                    status=resp.status,
                    content_type="application/json"
                )
        except Exception as e:
            provider["status"] = "error"
            return web.json_response(
                {"error": f"Provider error: {str(e)}"},
                status=502
            )


async def proxy_models(request):
    """Proxy models list request to a provider."""
    provider = await get_next_provider()
    if not provider:
        return web.json_response({"data": []})
    
    target_url = f"http://{provider['host']}:{provider['port']}/v1/models"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(target_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                response_body = await resp.read()
                return web.Response(
                    body=response_body,
                    status=resp.status,
                    content_type="application/json"
                )
        except Exception as e:
            return web.json_response({"data": []})


async def init_app():
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/api/v1/providers/register", register_provider)
    app.router.add_get("/api/v1/providers", list_providers)
    app.router.add_post("/v1/chat/completions", proxy_chat)
    app.router.add_post("/v1/completions", proxy_completions)
    app.router.add_get("/v1/models", proxy_models)
    return app


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"[Coordinator] Starting on port {port}")
    print(f"[Coordinator] OpenAI-compatible endpoint: http://localhost:{port}/v1/chat/completions")
    
    app = asyncio.run(init_app())
    web.run_app(app, host="0.0.0.0", port=port)
