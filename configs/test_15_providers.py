#!/usr/bin/env python3
"""
DiCAI 15-Provider Test Script

Auto-generated test for tinyllama-1.1b distributed across 15 providers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import grpc

import proto.dicai_pb2 as dicai_pb2
import proto.dicai_pb2_grpc as dicai_pb2_grpc


# Provider addresses
PROVIDERS = [
    "localhost:50051",  # p1 (layers 0-1)
    "localhost:50052",  # p2 (layers 2-3)
    "localhost:50053",  # p3 (layers 4-5)
    "localhost:50054",  # p4 (layers 6-7)
    "localhost:50055",  # p5 (layers 8-9)
    "localhost:50056",  # p6 (layers 10-11)
    "localhost:50057",  # p7 (layers 12-13)
    "localhost:50058",  # p8 (layers 14-14)
    "localhost:50059",  # p9 (layers 15-15)
    "localhost:50060",  # p10 (layers 16-16)
    "localhost:50061",  # p11 (layers 17-17)
    "localhost:50062",  # p12 (layers 18-18)
    "localhost:50063",  # p13 (layers 19-19)
    "localhost:50064",  # p14 (layers 20-20)
    "localhost:50065",  # p15 (layers 21-21)
]


def test_provider(address, input_ids=None, hidden_states=None, hidden_shape=None):
    """Test a single provider."""
    channel = grpc.insecure_channel(address)
    stub = dicai_pb2_grpc.ProviderServiceStub(channel)
    
    # Health check
    try:
        response = stub.health(dicai_pb2.HealthRequest(request_id="test"))
        print(f"  {{address}}: {{response.status}} ({{response.layers}})")
    except grpc.RpcError as e:
        print(f"  {{address}}: FAILED - {{e}}")
        return None
    
    # Process
    if hidden_states is not None:
        request = dicai_pb2.ProcessRequest(
            hidden_states=hidden_states.tobytes(),
            hidden_states_shape=hidden_shape,
            request_id="test"
        )
    else:
        request = dicai_pb2.ProcessRequest(
            input_ids=input_ids,
            request_id="test"
        )
    
    try:
        response = stub.process(request)
        if response.status == "success":
            return response
        else:
            print(f"  Error: {{response.error}}")
            return None
    except grpc.RpcError as e:
        print(f"  Process failed: {{e}}")
        return None


def test_end_to_end():
    """Test full chain of providers."""
    print(f"\n{{'='*60}}")
    print(f"Testing {len(PROVIDERS)}-Provider Chain")
    print(f"{{'='*60}}")
    
    # Start with input IDs
    input_ids = [1, 2, 3]
    hidden_states = None
    hidden_shape = None
    
    for i, address in enumerate(PROVIDERS):
        print(f"\n[Provider {i+1}] {{address}}")
        
        response = test_provider(address, input_ids, hidden_states, hidden_shape)
        
        if response is None:
            print(f"FAILED at provider {i+1}")
            return
        
        # Pass hidden states to next provider
        if response.hidden_states:
            hidden_states = np.frombuffer(response.hidden_states, dtype=np.float32)
            hidden_shape = list(response.hidden_states_shape)
            print(f"  Hidden states: {{hidden_shape}}")
        
        # Clear input_ids after first provider
        input_ids = None
    
    # Final logits
    if response.logits:
        logits_np = np.frombuffer(response.logits, dtype=np.float32)
        logits_shape = list(response.logits_shape)
        logits = torch.tensor(logits_np.reshape(logits_shape))
        predicted_token = torch.argmax(logits[0, -1]).item()
        
        print(f"\n{{'='*60}}")
        print(f"SUCCESS! Predicted token: {{predicted_token}}")
        print(f"{{'='*60}}")
    else:
        print("\nNo logits from final provider")


if __name__ == "__main__":
    test_end_to_end()
