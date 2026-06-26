#!/usr/bin/env python3
"""
DiCAI gRPC Client + HTTP Fallback

Tests both gRPC and HTTP providers.
"""

import argparse
import numpy as np
import torch

import grpc
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import proto.dicai_pb2 as dicai_pb2
import proto.dicai_pb2_grpc as dicai_pb2_grpc


def test_grpc_provider(address, input_ids=None, hidden_states=None, hidden_shape=None):
    """Test a gRPC provider."""
    print(f"\n{'='*60}")
    print(f"Testing gRPC Provider: {address}")
    print(f"{'='*60}")
    
    channel = grpc.insecure_channel(address)
    stub = dicai_pb2_grpc.ProviderServiceStub(channel)
    
    # Health check
    print("\n[1] Health Check")
    try:
        response = stub.health(dicai_pb2.HealthRequest(request_id="test-1"))
        print(f"  Status: {response.status}")
        print(f"  Provider: {response.provider_id}")
        print(f"  Layers: {response.layers}")
        print(f"  Memory: {response.memory_used_mb}/{response.memory_total_mb} MB")
    except grpc.RpcError as e:
        print(f"  FAILED: {e}")
        return
        
    # Process
    print("\n[2] Process Request")
    try:
        if hidden_states is not None:
            # Pass hidden states
            hidden_bytes = hidden_states.tobytes()
            request = dicai_pb2.ProcessRequest(
                hidden_states=hidden_bytes,
                hidden_states_shape=hidden_shape,
                request_id="test-2"
            )
        else:
            # Pass input IDs
            request = dicai_pb2.ProcessRequest(
                input_ids=input_ids,
                request_id="test-2"
            )
            
        response = stub.process(request)
        
        print(f"  Status: {response.status}")
        print(f"  Provider: {response.provider_id}")
        print(f"  Layers: {response.layers_processed}")
        
        if response.hidden_states:
            hidden_np = np.frombuffer(response.hidden_states, dtype=np.float32)
            hidden_shape = list(response.hidden_states_shape)
            print(f"  Hidden states shape: {hidden_shape}")
            
        if response.logits:
            logits_np = np.frombuffer(response.logits, dtype=np.float32)
            logits_shape = list(response.logits_shape)
            print(f"  Logits shape: {logits_shape}")
            
            # Get predicted token
            logits = torch.tensor(logits_np.reshape(logits_shape))
            predicted_token = torch.argmax(logits[0, -1]).item()
            print(f"  Predicted token: {predicted_token}")
            
        return response
        
    except grpc.RpcError as e:
        print(f"  FAILED: {e}")
        return


def test_end_to_end(p1_address, p2_address):
    """Test end-to-end inference with two providers."""
    print(f"\n{'='*60}")
    print(f"End-to-End Test: {p1_address} -> {p2_address}")
    print(f"{'='*60}")
    
    # Step 1: Provider 1 processes input_ids
    print("\n[Step 1] Provider 1 (layers 0-10)")
    response1 = test_grpc_provider(p1_address, input_ids=[1, 2, 3])
    
    if response1 is None or response1.status != "success":
        print("FAILED at Provider 1")
        return
        
    # Step 2: Provider 2 processes hidden_states
    print("\n[Step 2] Provider 2 (layers 11-21)")
    hidden_np = np.frombuffer(response1.hidden_states, dtype=np.float32)
    hidden_shape = list(response1.hidden_states_shape)
    
    response2 = test_grpc_provider(p2_address, hidden_states=hidden_np, hidden_shape=hidden_shape)
    
    if response2 is None or response2.status != "success":
        print("FAILED at Provider 2")
        return
        
    # Step 3: Get predicted token
    if response2.logits:
        logits_np = np.frombuffer(response2.logits, dtype=np.float32)
        logits_shape = list(response2.logits_shape)
        logits = torch.tensor(logits_np.reshape(logits_shape))
        predicted_token = torch.argmax(logits[0, -1]).item()
        
        print(f"\n{'='*60}")
        print(f"SUCCESS! Predicted token: {predicted_token}")
        print(f"{'='*60}")
    else:
        print("\nNo logits returned")


def main():
    parser = argparse.ArgumentParser(description="DiCAI gRPC Client")
    parser.add_argument("--p1", default="localhost:50051", help="Provider 1 address")
    parser.add_argument("--p2", default="localhost:50052", help="Provider 2 address")
    parser.add_argument("--test", choices=["health", "process", "e2e"], default="e2e", help="Test type")
    
    args = parser.parse_args()
    
    if args.test == "health":
        test_grpc_provider(args.p1)
        test_grpc_provider(args.p2)
    elif args.test == "process":
        test_grpc_provider(args.p1, input_ids=[1, 2, 3])
    else:
        test_end_to_end(args.p1, args.p2)


if __name__ == "__main__":
    main()
