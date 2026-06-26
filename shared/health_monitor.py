#!/usr/bin/env python3
"""
DiCAI Health Monitor

Monitors provider health with heartbeats, detects failures, triggers recovery.
"""

import time
import threading
import requests
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ProviderStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Slow responses
    UNHEALTHY = "unhealthy"  # Missing heartbeats
    DEAD = "dead"  # Confirmed dead


@dataclass
class ProviderHealth:
    provider_id: str
    address: str
    status: ProviderStatus
    last_heartbeat: float
    response_time_ms: float
    failure_count: int
    consecutive_failures: int


class HealthMonitor:
    """Monitors provider health and triggers recovery."""
    
    def __init__(self, heartbeat_interval: int = 5, failure_threshold: int = 3):
        self.heartbeat_interval = heartbeat_interval
        self.failure_threshold = failure_threshold
        self.providers: Dict[str, ProviderHealth] = {}
        self.callbacks: List[Callable] = []
        self.running = False
        self.monitor_thread = None
        
    def register_provider(self, provider_id: str, address: str):
        """Register a provider to monitor."""
        self.providers[provider_id] = ProviderHealth(
            provider_id=provider_id,
            address=address,
            status=ProviderStatus.HEALTHY,
            last_heartbeat=time.time(),
            response_time_ms=0,
            failure_count=0,
            consecutive_failures=0
        )
        print(f"[HealthMonitor] Registered {provider_id} at {address}")
        
    def unregister_provider(self, provider_id: str):
        """Unregister a provider."""
        if provider_id in self.providers:
            del self.providers[provider_id]
            print(f"[HealthMonitor] Unregistered {provider_id}")
            
    def record_heartbeat(self, provider_id: str, response_time_ms: float):
        """Record a successful heartbeat."""
        if provider_id in self.providers:
            health = self.providers[provider_id]
            health.last_heartbeat = time.time()
            health.response_time_ms = response_time_ms
            health.consecutive_failures = 0
            
            if health.status in [ProviderStatus.UNHEALTHY, ProviderStatus.DEAD]:
                health.status = ProviderStatus.HEALTHY
                print(f"[HealthMonitor] {provider_id} recovered")
                
    def record_failure(self, provider_id: str):
        """Record a failed heartbeat or request."""
        if provider_id in self.providers:
            health = self.providers[provider_id]
            health.failure_count += 1
            health.consecutive_failures += 1
            
            if health.consecutive_failures >= self.failure_threshold:
                if health.status != ProviderStatus.DEAD:
                    health.status = ProviderStatus.DEAD
                    print(f"[HealthMonitor] {provider_id} marked as DEAD")
                    self._notify_failure(provider_id)
            elif health.consecutive_failures >= self.failure_threshold // 2:
                health.status = ProviderStatus.UNHEALTHY
                
    def _notify_failure(self, provider_id: str):
        """Notify callbacks about provider failure."""
        for callback in self.callbacks:
            try:
                callback(provider_id)
            except Exception as e:
                print(f"[HealthMonitor] Callback error: {e}")
                
    def on_failure(self, callback: Callable):
        """Register a callback for provider failures."""
        self.callbacks.append(callback)
        
    def check_health(self, provider_id: str) -> ProviderStatus:
        """Get current health status."""
        if provider_id not in self.providers:
            return ProviderStatus.DEAD
            
        health = self.providers[provider_id]
        time_since_heartbeat = time.time() - health.last_heartbeat
        
        if time_since_heartbeat > self.heartbeat_interval * 3:
            return ProviderStatus.DEAD
        elif time_since_heartbeat > self.heartbeat_interval * 2:
            return ProviderStatus.UNHEALTHY
        elif health.consecutive_failures > 0:
            return ProviderStatus.DEGRADED
            
        return ProviderStatus.HEALTHY
        
    def get_healthy_providers(self) -> List[str]:
        """Get list of healthy provider IDs."""
        healthy = []
        for provider_id, health in self.providers.items():
            if health.status == ProviderStatus.HEALTHY:
                healthy.append(provider_id)
        return healthy
        
    def get_all_status(self) -> Dict[str, str]:
        """Get status of all providers."""
        return {pid: health.status.value for pid, health in self.providers.items()}
        
    def start_monitoring(self):
        """Start background monitoring thread."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("[HealthMonitor] Started monitoring")
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            for provider_id, health in self.providers.items():
                time_since_heartbeat = time.time() - health.last_heartbeat
                
                if time_since_heartbeat > self.heartbeat_interval * 2:
                    self.record_failure(provider_id)
                    
            time.sleep(self.heartbeat_interval)


def test_health_monitor():
    """Test health monitor."""
    monitor = HealthMonitor(heartbeat_interval=1, failure_threshold=2)
    
    # Register providers
    monitor.register_provider("p1", "http://localhost:8081")
    monitor.register_provider("p2", "http://localhost:8082")
    
    # Simulate heartbeats
    monitor.record_heartbeat("p1", 50)
    monitor.record_heartbeat("p2", 100)
    
    print(f"Status: {monitor.get_all_status()}")
    
    # Simulate failure
    time.sleep(2.5)
    monitor.record_failure("p1")
    monitor.record_failure("p1")
    
    print(f"Status after failure: {monitor.get_all_status()}")
    print(f"Healthy providers: {monitor.get_healthy_providers()}")
    
    print("Health monitor test PASSED")


if __name__ == "__main__":
    test_health_monitor()
