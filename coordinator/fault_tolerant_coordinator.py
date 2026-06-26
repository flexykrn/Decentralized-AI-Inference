#!/usr/bin/env python3
"""
DiCAI Coordinator - Fault Tolerant Inference Router

Routes inference through provider chain with retry and fallback.
"""

import argparse
import json
import time
import numpy as np
import torch

import grpc
from concurrent import futures

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import proto.dicai_pb2 as dicai_pb2
import proto.dicai_pb2_grpc as dicai_pb2_grpc


class CircuitBreaker:
    """Simple circuit breaker for provider calls."""
    
    def __init__(self, failure_threshold=3, recovery_timeout=10):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
                
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
            raise e


class FaultTolerantClient:
    """gRPC client with retry and circuit breaker."""
    
    def __init__(self, address):
        self.address = address
        self.channel = None
        self.stub = None
        self.circuit_breaker = CircuitBreaker()
        self.connect()
        
    def connect(self):
        """Connect to provider."""
        try:
            self.channel = grpc.insecure_channel(self.address)
            grpc.channel_ready_future(self.channel).result(timeout=5)
            self.stub = dicai_pb2_grpc.ProviderServiceStub(self.channel)
            return True
        except Exception as e:
            print(f"Failed to connect to {self.address}: {e}")
            return False
            
    def process(self, request):
        """Process with retry and circuit breaker."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self.circuit_breaker.call(self.stub.process, request)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = 1 * (2 ** attempt)
                print(f"Retry {attempt + 1} for {self.address} after {delay}s")
                time.sleep(delay)
        return None
        
    def health(self, request):
        """Health check with retry."""
        try:
            return self.circuit_breaker.call(self.stub.health, request)
        except Exception as e:
            return None


class Coordinator:
    """Fault-tolerant coordinator for distributed inference."""
    
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.providers = {}
        self.provider_chain = []
        self.connect_providers()
        
    def load_config(self, config_file):
        """Load provider configuration."""
        with open(config_file) as f:
            return json.load(f)
            
    def connect_providers(self):
        """Connect to all providers."""
        print("[Coordinator] Connecting to providers...")
        
        for shard in self.config['shards']:
            provider_id = shard['provider_id']
            address = f"{shard['host']}:{shard['port']}"
            
            client = FaultTolerantClient(address)
            if client.stub:
                self.providers[provider_id] = client
                self.provider_chain.append(provider_id)
                print(f"  [OK] {provider_id} at {address}")
            else:
                print(f"  [FAIL] {provider_id} at {address}")
                
        print(f"[Coordinator] Connected to {len(self.providers)}/{len(self.config['shards'])} providers")
        
    def health_check_all(self):
        """Check health of all providers."""
        print("\n[Coordinator] Health Check")
        healthy = []
        
        for provider_id, client in self.providers.items():
            try:
                response = client.health(dicai_pb2.HealthRequest(request_id="health-check"))
                if response:
                    print(f"  [OK] {provider_id}: {response.status} ({response.layers})")
                    healthy.append(provider_id)
                else:
                    print(f"  [FAIL] {provider_id}: No response")
            except Exception as e:
                print(f"  [FAIL] {provider_id}: {e}")
                
        return healthy
        
    def inference(self, input_ids):
        """Run inference through provider chain."""
        print(f"\n[Coordinator] Starting inference with {len(self.provider_chain)} providers")
        
        request_id = f"req-{int(time.time() * 1000)}"
        hidden_states = None
        hidden_shape = None
        
        for i, provider_id in enumerate(self.provider_chain):
            print(f"\n[Step {i+1}] Provider: {provider_id}")
            
            client = self.providers.get(provider_id)
            if not client:
                print(f"  [SKIP] Provider {provider_id} not available")
                continue
                
            # Build request
            if hidden_states is not None:
                request = dicai_pb2.ProcessRequest(
                    hidden_states=hidden_states.tobytes(),
                    hidden_states_shape=hidden_shape,
                    request_id=request_id
                )
            else:
                request = dicai_pb2.ProcessRequest(
                    input_ids=input_ids,
                    request_id=request_id
                )
                
            # Send to provider
            try:
                response = client.process(request)
                
                if response.status == "success":
                    print(f"  [OK] Layers: {response.layers_processed}")
                    
                    if response.hidden_states:
                        hidden_states = np.frombuffer(response.hidden_states, dtype=np.float32)
                        hidden_shape = list(response.hidden_states_shape)
                        print(f"  Hidden states: {hidden_shape}")
                        
                    if response.logits:
                        logits_np = np.frombuffer(response.logits, dtype=np.float32)
                        logits_shape = list(response.logits_shape)
                        logits = torch.tensor(logits_np.reshape(logits_shape))
                        predicted_token = torch.argmax(logits[0, -1]).item()
                        
                        print(f"\n{'='*60}")
                        print(f"SUCCESS! Predicted token: {predicted_token}")
                        print(f"{'='*60}")
                        return predicted_token
                else:
                    print(f"  [ERROR] {response.error}")
                    
            except Exception as e:
                print(f"  [FAIL] {e}")
                
        print("\n[Coordinator] Inference failed")
        return None
        
    def close(self):
        """Close all connections."""
        for client in self.providers.values():
            if client.channel:
                client.channel.close()


def main():
    parser = argparse.ArgumentParser(description="DiCAI Fault-Tolerant Coordinator")
    parser.add_argument("--config", default="configs/15_providers.json", help="Provider config")
    parser.add_argument("--test", action="store_true", help="Run inference test")
    
    args = parser.parse_args()
    
    coordinator = Coordinator(args.config)
    
    # Health check
    healthy = coordinator.health_check_all()
    print(f"\nHealthy providers: {len(healthy)}/{len(coordinator.provider_chain)}")
    
    if args.test and healthy:
        # Test inference
        coordinator.inference([1, 2, 3])
    
    coordinator.close()


if __name__ == "__main__":
    main()
