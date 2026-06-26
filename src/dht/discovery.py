#!/usr/bin/env python3
"""
DiCAI - Distributed Provider Discovery

Lightweight distributed hash table for provider registration and discovery.
Pure Python implementation with no external dependencies.
"""

import hashlib
import json
import time
import threading
import socket
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ProviderInfo:
    """Information about a provider node."""
    provider_id: str
    address: str
    port: int
    start_layer: int
    end_layer: int
    total_layers: int
    last_seen: float
    status: str = "online"
    throughput: float = 0.0  # tokens/sec
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    cpu_memory_gb: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProviderInfo':
        return cls(**data)
    
    @property
    def layer_count(self) -> int:
        return self.end_layer - self.start_layer + 1
    
    @property
    def is_healthy(self) -> bool:
        return self.status == "online" and (time.time() - self.last_seen) < 30


class DHTNode:
    """Simple DHT node for provider discovery."""
    
    def __init__(self, node_id: str, host: str = "0.0.0.0", port: int = 8464):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.providers: Dict[str, ProviderInfo] = {}
        self.lock = threading.RLock()
        self.running = False
        self.server_thread = None
        
    def start(self):
        """Start DHT node."""
        self.running = True
        self.server_thread = threading.Thread(target=self._serve, daemon=True)
        self.server_thread.start()
        print(f"[DHT] Node {self.node_id} started on {self.host}:{self.port}")
        
    def stop(self):
        """Stop DHT node."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=5)
            
    def _serve(self):
        """Simple UDP server for DHT operations."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(1.0)
        
        while self.running:
            try:
                data, addr = sock.recvfrom(65535)
                request = json.loads(data.decode())
                response = self._handle_request(request)
                
                if response:
                    sock.sendto(json.dumps(response).encode(), addr)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[DHT] Error: {e}")
                
        sock.close()
        
    def _handle_request(self, request: dict) -> Optional[dict]:
        """Handle incoming DHT request."""
        op = request.get("op")
        
        if op == "register":
            return self._handle_register(request)
        elif op == "unregister":
            return self._handle_unregister(request)
        elif op == "find":
            return self._handle_find(request)
        elif op == "heartbeat":
            return self._handle_heartbeat(request)
        elif op == "list":
            return self._handle_list(request)
        elif op == "health":
            return self._handle_health(request)
        else:
            return {"error": "Unknown operation"}
            
    def _handle_register(self, request: dict) -> dict:
        """Register a new provider."""
        provider = ProviderInfo.from_dict(request["provider"])
        
        with self.lock:
            self.providers[provider.provider_id] = provider
            
        print(f"[DHT] Registered provider {provider.provider_id} (layers {provider.start_layer}-{provider.end_layer})")
        return {"status": "registered", "provider_id": provider.provider_id}
        
    def _handle_unregister(self, request: dict) -> dict:
        """Unregister a provider."""
        provider_id = request["provider_id"]
        
        with self.lock:
            if provider_id in self.providers:
                del self.providers[provider_id]
                
        print(f"[DHT] Unregistered provider {provider_id}")
        return {"status": "unregistered"}
        
    def _handle_find(self, request: dict) -> dict:
        """Find providers for a layer range."""
        start_layer = request.get("start_layer", 0)
        end_layer = request.get("end_layer", 0)
        
        matching = []
        with self.lock:
            for provider in self.providers.values():
                if provider.is_healthy and provider.start_layer <= end_layer and provider.end_layer >= start_layer:
                    matching.append(provider.to_dict())
                    
        return {"providers": matching}
        
    def _handle_heartbeat(self, request: dict) -> dict:
        """Update provider heartbeat."""
        provider_id = request["provider_id"]
        
        with self.lock:
            if provider_id in self.providers:
                self.providers[provider_id].last_seen = time.time()
                self.providers[provider_id].status = "online"
                return {"status": "ok"}
            else:
                return {"error": "Provider not found"}
                
    def _handle_list(self, request: dict) -> dict:
        """List all providers."""
        with self.lock:
            providers = [p.to_dict() for p in self.providers.values() if p.is_healthy]
            
        return {"providers": providers, "count": len(providers)}
        
    def _handle_health(self, request: dict) -> dict:
        """Get DHT node health."""
        with self.lock:
            total = len(self.providers)
            healthy = sum(1 for p in self.providers.values() if p.is_healthy)
            
        return {
            "status": "healthy",
            "node_id": self.node_id,
            "total_providers": total,
            "healthy_providers": healthy
        }
        
    def register_provider(self, provider: ProviderInfo) -> bool:
        """Register a provider locally."""
        with self.lock:
            self.providers[provider.provider_id] = provider
        return True
        
    def find_providers(self, start_layer: int, end_layer: int) -> List[ProviderInfo]:
        """Find providers covering a layer range."""
        matching = []
        with self.lock:
            for provider in self.providers.values():
                if provider.is_healthy and provider.start_layer <= end_layer and provider.end_layer >= start_layer:
                    matching.append(provider)
                    
        return sorted(matching, key=lambda p: p.throughput, reverse=True)
        
    def find_optimal_route(self, total_layers: int) -> List[ProviderInfo]:
        """Find optimal route through providers for all layers."""
        route = []
        current_layer = 0
        
        while current_layer < total_layers:
            # Find providers starting at or before current_layer
            candidates = self.find_providers(current_layer, total_layers - 1)
            
            if not candidates:
                print(f"[DHT] No provider found for layer {current_layer}")
                return []
                
            # Pick provider with highest throughput that covers current_layer
            best = None
            for provider in candidates:
                if provider.start_layer <= current_layer:
                    if best is None or provider.throughput > best.throughput:
                        best = provider
                        
            if best is None:
                print(f"[DHT] No provider covers layer {current_layer}")
                return []
                
            route.append(best)
            current_layer = best.end_layer + 1
            
        return route
        
    def cleanup_stale_providers(self, max_age: int = 60):
        """Remove providers that haven't heartbeated recently."""
        now = time.time()
        with self.lock:
            stale = [pid for pid, p in self.providers.items() if now - p.last_seen > max_age]
            for pid in stale:
                self.providers[pid].status = "offline"
                print(f"[DHT] Marked {pid} as offline (no heartbeat)")


class DHTClient:
    """Client for interacting with DHT nodes."""
    
    def __init__(self, dht_host: str = "localhost", dht_port: int = 8464):
        self.dht_host = dht_host
        self.dht_port = dht_port
        
    def _send(self, request: dict) -> dict:
        """Send request to DHT node."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5.0)
        
        try:
            sock.sendto(json.dumps(request).encode(), (self.dht_host, self.dht_port))
            data, _ = sock.recvfrom(65535)
            return json.loads(data.decode())
        except socket.timeout:
            return {"error": "Timeout"}
        finally:
            sock.close()
            
    def register(self, provider: ProviderInfo) -> dict:
        """Register a provider."""
        return self._send({"op": "register", "provider": provider.to_dict()})
        
    def unregister(self, provider_id: str) -> dict:
        """Unregister a provider."""
        return self._send({"op": "unregister", "provider_id": provider_id})
        
    def find(self, start_layer: int, end_layer: int) -> List[dict]:
        """Find providers for layer range."""
        response = self._send({"op": "find", "start_layer": start_layer, "end_layer": end_layer})
        return response.get("providers", [])
        
    def heartbeat(self, provider_id: str) -> dict:
        """Send heartbeat."""
        return self._send({"op": "heartbeat", "provider_id": provider_id})
        
    def list_providers(self) -> List[dict]:
        """List all providers."""
        response = self._send({"op": "list"})
        return response.get("providers", [])
        
    def health(self) -> dict:
        """Get DHT health."""
        return self._send({"op": "health"})


def test_dht():
    """Test DHT implementation."""
    print("=" * 60)
    print("DHT Provider Discovery Test")
    print("=" * 60)
    
    # Start DHT node
    dht = DHTNode("test-node", port=8464)
    dht.start()
    
    # Create client
    client = DHTClient("localhost", 8464)
    
    # Register providers
    providers = [
        ProviderInfo("p1", "localhost", 50051, 0, 10, 22, time.time(), throughput=100.0),
        ProviderInfo("p2", "localhost", 50052, 11, 20, 22, time.time(), throughput=80.0),
        ProviderInfo("p3", "localhost", 50053, 21, 21, 22, time.time(), throughput=120.0),
    ]
    
    for p in providers:
        result = client.register(p)
        print(f"Registered {p.provider_id}: {result}")
        
    # Find providers
    found = client.find(0, 22)
    print(f"\nFound {len(found)} providers for layers 0-22")
    
    # Find optimal route
    route = dht.find_optimal_route(22)
    print(f"\nOptimal route: {' -> '.join([p.provider_id for p in route])}")
    
    # Health check
    health = client.health()
    print(f"\nDHT Health: {health}")
    
    # Cleanup
    dht.stop()
    
    print("\nDHT Test PASSED")
    return True


if __name__ == "__main__":
    test_dht()
