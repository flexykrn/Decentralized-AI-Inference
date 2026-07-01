#!/usr/bin/env python3
"""
DiCAI - Coordinator

Manages provider registry, calculates layer assignments, builds inference routes.
Handles dynamic rebalancing when providers join or leave.

Usage:
    python -m dicai.coordinator --port 8464
"""

import os
import sys
import time
import json
import argparse
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dht.discovery import DHTNode, DHTClient, ProviderInfo


@dataclass
class LayerAssignment:
    """Layer assignment for a provider."""
    provider_id: str
    start_layer: int
    end_layer: int
    version: int = 1


class Coordinator:
    """Central coordinator for distributed inference."""
    
    def __init__(self, port: int = 8464, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.dht = DHTNode("coordinator", host=host, port=port)
        
        self.assignments: Dict[str, LayerAssignment] = {}
        self.assignment_version = 0
        self.total_layers = 22  # Default for TinyLlama
        
        self.app = FastAPI(title="DiCAI Coordinator")
        self._setup_routes()
        
        self.running = False
        self.rebalance_thread = None
        
    def _setup_routes(self):
        """Setup HTTP endpoints."""
        
        @self.app.post("/assign")
        async def assign(request: Dict):
            """Request layer assignment for a provider (legacy body)."""
            provider_id = request.get("provider_id")
            capacity_score = request.get("capacity_score", 1.0)
            
            assignment = self._calculate_assignment(provider_id, capacity_score)
            return asdict(assignment)
            
        @self.app.post("/assign/{provider_id}")
        async def assign_by_id(provider_id: str, request: Dict):
            """Request layer assignment for a provider by URL."""
            capacity_score = request.get("capacity_score", 1.0)
            
            assignment = self._calculate_assignment(provider_id, capacity_score)
            return asdict(assignment)
            
        @self.app.get("/route")
        async def get_route():
            """Get current inference route."""
            route = self._build_route()
            return {
                "route": [asdict(a) for a in route],
                "version": self.assignment_version,
                "total_layers": self.total_layers
            }
            
        @self.app.get("/providers")
        async def list_providers():
            """List all registered providers."""
            providers = self.dht.list_providers()
            return {"providers": providers, "count": len(providers)}
            
        @self.app.post("/rebalance")
        async def rebalance():
            """Trigger manual rebalancing."""
            self._rebalance_all()
            return {"status": "rebalanced", "version": self.assignment_version}
            
        @self.app.get("/health")
        async def health():
            """Health check."""
            return {
                "status": "healthy",
                "providers": len(self.dht.providers),
                "assignments": len(self.assignments),
                "version": self.assignment_version
            }

        @self.app.get("/status")
        async def status():
            """Detailed coordinator status."""
            route = self._build_route()
            return {
                "status": "healthy",
                "providers": len(self.dht.providers),
                "assignments": [asdict(a) for a in self.assignments.values()],
                "route": [asdict(a) for a in route],
                "version": self.assignment_version,
                "total_layers": self.total_layers
            }
            
    def _calculate_assignment(self, provider_id: str, capacity_score: float) -> LayerAssignment:
        """Calculate layer assignment for a provider."""
        # Get all providers from DHT
        all_providers = []
        with self.dht.lock:
            all_providers = [p.to_dict() for p in self.dht.providers.values() if p.is_healthy]
        
        if not all_providers:
            # First provider gets all layers
            return LayerAssignment(provider_id, 0, self.total_layers - 1)
            
        # Calculate total capacity including the new provider
        total_capacity = sum(p.get("capacity_score", 1.0) for p in all_providers) + capacity_score
        
        # Calculate proportional share
        layer_share = int((capacity_score / total_capacity) * self.total_layers)
        layer_share = max(1, layer_share)
        
        # Find first gap in current assignments
        assigned_ranges = sorted([(a.start_layer, a.end_layer) for a in self.assignments.values()])
        
        start_layer = 0
        for s, e in assigned_ranges:
            if start_layer < s:
                break
            start_layer = e + 1
            
        # Don't exceed remaining layers
        remaining = self.total_layers - start_layer
        if remaining <= 0:
            start_layer = self.total_layers - 1
            remaining = 1
        
        end_layer = min(start_layer + min(layer_share, remaining) - 1, self.total_layers - 1)
        
        assignment = LayerAssignment(provider_id, start_layer, end_layer, self.assignment_version)
        self.assignments[provider_id] = assignment
        
        return assignment
        
    def _build_route(self) -> List[LayerAssignment]:
        """Build optimal inference route from DHT providers."""
        # Get providers from DHT
        providers = []
        with self.dht.lock:
            providers = [p for p in self.dht.providers.values() if p.is_healthy]
        
        if not providers:
            return []
        
        # Sort by start_layer, then prefer contiguous providers with highest throughput
        providers = sorted(providers, key=lambda p: (p.start_layer, -p.throughput))
        
        # Build non-overlapping contiguous route
        route = []
        current_layer = 0
        
        for provider in providers:
            if provider.start_layer > current_layer:
                print(f"[Coordinator] Gap detected: layers {current_layer}-{provider.start_layer - 1}")
                break
            if provider.start_layer <= current_layer <= provider.end_layer:
                route.append(LayerAssignment(
                    provider_id=provider.provider_id,
                    start_layer=current_layer,
                    end_layer=provider.end_layer,
                    version=self.assignment_version
                ))
                current_layer = provider.end_layer + 1
                if current_layer >= self.total_layers:
                    break
        
        if current_layer < self.total_layers:
            print(f"[Coordinator] Gap detected: layers {current_layer}-{self.total_layers - 1}")
        
        return route
        
    def _rebalance_all(self):
        """Recalculate all layer assignments."""
        print("[Coordinator] Rebalancing...")
        
        # Get all providers from DHT
        providers = []
        with self.dht.lock:
            providers = [p.to_dict() for p in self.dht.providers.values() if p.is_healthy]
        
        if not providers:
            return
            
        # Calculate total capacity
        total_capacity = sum(p.get("capacity_score", 1.0) for p in providers)
        
        # Clear old assignments
        self.assignments.clear()
        self.assignment_version += 1
        
        # Assign layers proportionally
        current_layer = 0
        for provider_data in providers:
            provider_id = provider_data["provider_id"]
            capacity = provider_data.get("capacity_score", 1.0)
            
            layer_count = int((capacity / total_capacity) * self.total_layers)
            layer_count = max(1, layer_count)
            
            start_layer = current_layer
            end_layer = min(current_layer + layer_count - 1, self.total_layers - 1)
            
            self.assignments[provider_id] = LayerAssignment(
                provider_id, start_layer, end_layer, self.assignment_version
            )
            
            current_layer = end_layer + 1
            
        # Notify providers of new assignments
        self._notify_providers()
        
        print(f"[Coordinator] Rebalanced to version {self.assignment_version}")
        
    def _notify_providers(self):
        """Notify providers of new assignments."""
        for assignment in self.assignments.values():
            # In real implementation, send HTTP POST to provider's /reload endpoint
            print(f"[Coordinator] Notify {assignment.provider_id}: layers {assignment.start_layer}-{assignment.end_layer}")
            
    def _rebalance_loop(self):
        """Periodic rebalancing check."""
        while self.running:
            # Check for new providers or departed providers
            current_providers = set()
            with self.dht.lock:
                current_providers = set(self.dht.providers.keys())
            assigned_providers = set(self.assignments.keys())
            
            if current_providers != assigned_providers:
                print("[Coordinator] Provider change detected, rebalancing...")
                self._rebalance_all()
                
            time.sleep(10)
            
    def start(self):
        """Start coordinator."""
        print(f"[Coordinator] Starting on {self.host}:{self.port}")
        
        # Start DHT
        self.dht.start()
        
        # Start rebalancing loop
        self.running = True
        self.rebalance_thread = threading.Thread(target=self._rebalance_loop, daemon=True)
        self.rebalance_thread.start()
        
        # Start HTTP server
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")
        
    def stop(self):
        """Stop coordinator."""
        self.running = False
        self.dht.stop()
        if self.rebalance_thread:
            self.rebalance_thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description="DiCAI Coordinator")
    parser.add_argument("--port", type=int, default=8464, help="Coordinator port")
    parser.add_argument("--host", default="0.0.0.0", help="Coordinator host")
    
    args = parser.parse_args()
    
    coordinator = Coordinator(port=args.port, host=args.host)
    
    try:
        coordinator.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        coordinator.stop()


if __name__ == "__main__":
    main()
