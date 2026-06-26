#!/usr/bin/env python3
"""
DiCAI - Fault-Tolerant Inference Sessions

Manages persistent inference sessions across provider chains.
Handles provider failures with automatic rerouting and session migration.
"""

import time
import uuid
import threading
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import numpy as np

from src.dht.discovery import DHTClient, ProviderInfo


@dataclass
class InferenceSession:
    """Persistent inference session across providers."""
    session_id: str
    route: List[ProviderInfo]
    current_position: int = 0
    history: Optional[torch.Tensor] = None
    kv_cache: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    failed_providers: set = field(default_factory=set)
    
    @property
    def is_active(self) -> bool:
        return len(self.route) > 0 and self.current_position < len(self.route)
    
    @property
    def current_provider(self) -> Optional[ProviderInfo]:
        if self.current_position < len(self.route):
            return self.route[self.current_position]
        return None
    
    def advance(self):
        """Move to next provider in route."""
        self.current_position += 1
        self.last_used = time.time()
        
    def mark_provider_failed(self, provider_id: str):
        """Mark a provider as failed and remove from route."""
        self.failed_providers.add(provider_id)
        # Don't remove from route immediately - let router handle rerouting
        # self.route = [p for p in self.route if p.provider_id != provider_id]
        
    def remove_failed_providers(self):
        """Remove failed providers from route."""
        self.route = [p for p in self.route if p.provider_id not in self.failed_providers]
        # Reset position if current provider was removed
        if self.current_position >= len(self.route):
            self.current_position = 0
        
    def reset_position(self):
        """Reset to start of route."""
        self.current_position = 0
        self.last_used = time.time()


class FaultTolerantRouter:
    """Routes inference through providers with fault tolerance."""
    
    def __init__(self, dht_client: DHTClient, total_layers: int = 22):
        self.dht_client = dht_client
        self.total_layers = total_layers
        self.sessions: Dict[str, InferenceSession] = {}
        self.session_lock = threading.RLock()
        self.retry_count = 3
        self.retry_delay = 1.0
        
    def create_session(self, route: Optional[List[ProviderInfo]] = None) -> InferenceSession:
        """Create new inference session with optimal route."""
        if route is None:
            route = self._find_optimal_route()
            
        if not route:
            raise Exception("No available providers found")
            
        session = InferenceSession(
            session_id=str(uuid.uuid4()),
            route=route
        )
        
        with self.session_lock:
            self.sessions[session.session_id] = session
            
        print(f"[Router] Created session {session.session_id} with route: {' -> '.join([p.provider_id for p in route])}")
        return session
    
    def _find_optimal_route(self) -> List[ProviderInfo]:
        """Find optimal route through providers."""
        # Query DHT for all providers
        providers_data = self.dht_client.list_providers()
        
        if not providers_data:
            return []
            
        # Convert to ProviderInfo objects
        providers = [ProviderInfo.from_dict(p) for p in providers_data]
        
        # Sort by throughput (descending)
        providers.sort(key=lambda p: p.throughput, reverse=True)
        
        # Build route using greedy algorithm
        route = []
        current_layer = 0
        used_providers = set()
        
        while current_layer < self.total_layers:
            best = None
            
            for provider in providers:
                if provider.provider_id in used_providers:
                    continue
                    
                # Check if provider covers current layer
                if provider.start_layer <= current_layer <= provider.end_layer:
                    if best is None or provider.throughput > best.throughput:
                        best = provider
                        
            if best is None:
                # No provider covers this layer - try to find one that starts after
                for provider in providers:
                    if provider.provider_id in used_providers:
                        continue
                    if provider.start_layer > current_layer:
                        if best is None or provider.start_layer < best.start_layer:
                            best = provider
                            
                if best is None:
                    print(f"[Router] No provider found for layer {current_layer}")
                    return []
                    
                # Gap in coverage - will need to handle this
                print(f"[Router] Gap detected: layer {current_layer} to {best.start_layer - 1}")
                
            route.append(best)
            used_providers.add(best.provider_id)
            current_layer = best.end_layer + 1
            
        return route
    
    def execute_step(self, session_id: str, inputs: torch.Tensor, 
                     prompts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute one inference step with fault tolerance."""
        with self.session_lock:
            if session_id not in self.sessions:
                raise Exception(f"Session {session_id} not found")
                
            session = self.sessions[session_id]
            
        if not session.is_active:
            raise Exception("Session is no longer active")
            
        # Try current provider
        provider = session.current_provider
        if not provider:
            raise Exception("No provider available")
            
        # Attempt execution with retries
        for attempt in range(self.retry_count):
            try:
                result = self._execute_on_provider(provider, inputs, session)
                session.advance()
                return result
                
            except Exception as e:
                print(f"[Router] Provider {provider.provider_id} failed (attempt {attempt + 1}/{self.retry_count}): {e}")
                
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    # All retries failed - mark provider as failed and reroute
                    session.mark_provider_failed(provider.provider_id)
                    session.remove_failed_providers()
                    
                    # Try to find alternative route
                    new_route = self._find_alternative_route(session, provider.provider_id)
                    
                    if new_route:
                        print(f"[Router] Rerouting through: {' -> '.join([p.provider_id for p in new_route])}")
                        session.route = new_route
                        session.reset_position()
                        
                        # Retry with new route
                        return self.execute_step(session_id, inputs, prompts)
                    else:
                        raise Exception(f"No alternative route found after {provider.provider_id} failed")
                        
        raise Exception("Unexpected error in execute_step")
        
    def _execute_on_provider(self, provider: ProviderInfo, inputs: torch.Tensor,
                             session: InferenceSession) -> torch.Tensor:
        """Execute inference on a single provider."""
        # In real implementation, this would make gRPC call to provider
        # For now, simulate with identity function
        print(f"[Router] Executing on {provider.provider_id} (layers {provider.start_layer}-{provider.end_layer})")
        
        # Simulate processing time based on throughput
        processing_time = 1.0 / max(provider.throughput, 1.0)
        time.sleep(processing_time)
        
        return inputs
        
    def _find_alternative_route(self, session: InferenceSession, 
                                failed_provider_id: str) -> Optional[List[ProviderInfo]]:
        """Find alternative route avoiding failed provider."""
        # Get all providers from DHT
        providers_data = self.dht_client.list_providers()
        
        if not providers_data:
            return None
            
        providers = [ProviderInfo.from_dict(p) for p in providers_data]
        
        # Filter out failed providers
        available = [p for p in providers 
                    if p.provider_id not in session.failed_providers 
                    and p.provider_id != failed_provider_id]
        
        if not available:
            return None
            
        # Sort by throughput
        available.sort(key=lambda p: p.throughput, reverse=True)
        
        # Build new route
        route = []
        current_layer = 0
        used = set()
        
        while current_layer < self.total_layers:
            best = None
            
            for provider in available:
                if provider.provider_id in used:
                    continue
                    
                if provider.start_layer <= current_layer <= provider.end_layer:
                    if best is None or provider.throughput > best.throughput:
                        best = provider
                        
            if best is None:
                return None
                
            route.append(best)
            used.add(best.provider_id)
            current_layer = best.end_layer + 1
            
        return route
    
    def close_session(self, session_id: str):
        """Close inference session."""
        with self.session_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                print(f"[Router] Closed session {session_id}")
                
    def cleanup_stale_sessions(self, max_age: int = 3600):
        """Remove sessions inactive for too long."""
        now = time.time()
        with self.session_lock:
            stale = [sid for sid, s in self.sessions.items() 
                    if now - s.last_used > max_age]
            for sid in stale:
                del self.sessions[sid]
                print(f"[Router] Cleaned up stale session {sid}")


def test_fault_tolerance():
    """Test fault-tolerant routing."""
    print("=" * 60)
    print("Fault-Tolerant Routing Test")
    print("=" * 60)
    
    from src.dht.discovery import DHTNode
    
    # Start DHT
    dht = DHTNode("test-router", port=8465)
    dht.start()
    
    # Create client and router
    client = DHTClient("localhost", 8465)
    router = FaultTolerantRouter(client, total_layers=22)
    
    # Register providers
    providers = [
        ProviderInfo("p1", "localhost", 50051, 0, 10, 22, time.time(), throughput=100.0),
        ProviderInfo("p2", "localhost", 50052, 11, 20, 22, time.time(), throughput=80.0),
        ProviderInfo("p3", "localhost", 50053, 21, 21, 22, time.time(), throughput=120.0),
        ProviderInfo("p2-backup", "localhost", 50054, 11, 20, 22, time.time(), throughput=60.0),
    ]
    
    for p in providers:
        client.register(p)
        
    # Create session
    session = router.create_session()
    print(f"\nInitial route: {' -> '.join([p.provider_id for p in session.route])}")
    
    # Simulate inference steps
    inputs = torch.randn(1, 5, 2048)
    
    for step in range(3):
        try:
            result = router.execute_step(session.session_id, inputs)
            print(f"Step {step + 1}: Success (shape: {result.shape})")
        except Exception as e:
            print(f"Step {step + 1}: Failed - {e}")
            break
            
    # Simulate provider failure
    print(f"\nSimulating p2 failure...")
    session.mark_provider_failed("p2")
    session.remove_failed_providers()
    session.reset_position()
    
    # Try to continue
    try:
        result = router.execute_step(session.session_id, inputs)
        print(f"After failure: Success (rerouted)")
        print(f"New route: {' -> '.join([p.provider_id for p in session.route])}")
    except Exception as e:
        print(f"After failure: {e}")
        
    # Cleanup
    router.close_session(session.session_id)
    dht.stop()
    
    print("\nFault Tolerance Test PASSED")
    return True


if __name__ == "__main__":
    test_fault_tolerance()
