#!/usr/bin/env python3
"""
DiCAI 15+ Provider Configuration Generator

Generates shard configs for large models distributed across many providers.
Uses 1.1B model for testing (simulates 70B architecture with fewer layers).
"""

import argparse
import json
import os
import math


def calculate_shards(num_providers, total_layers, model_size_gb, precision="Q4_K_M"):
    """Calculate shard distribution for N providers."""
    
    # Memory per layer (approximate)
    if precision == "Q4_K_M":
        bytes_per_param = 0.5  # 4-bit quantized
    elif precision == "Q8_0":
        bytes_per_param = 1.0  # 8-bit
    elif precision == "BF16":
        bytes_per_param = 2.0  # 16-bit
    else:
        bytes_per_param = 0.5
    
    # Calculate base layers per provider and remainder
    base_layers = total_layers // num_providers
    remainder = total_layers % num_providers
    
    shards = []
    start_layer = 0
    
    for i in range(num_providers):
        # First 'remainder' providers get 1 extra layer
        extra = 1 if i < remainder else 0
        num_layers = base_layers + extra
        
        if num_layers > 0:
            end_layer = start_layer + num_layers - 1
            
            # Calculate approximate shard size
            shard_size_gb = (model_size_gb / total_layers) * num_layers
            
            shard = {
                "provider_id": f"p{i+1}",
                "start_layer": start_layer,
                "end_layer": end_layer,
                "num_layers": num_layers,
                "approx_size_gb": round(shard_size_gb, 2),
                "port": 50051 + i,
                "host": "localhost"
            }
            shards.append(shard)
            
            start_layer = end_layer + 1
        else:
            # No layers for this provider - skip
            break
        
    return shards


def generate_config(num_providers, model_name="tinyllama-1.1b", total_layers=22, 
                    model_size_gb=4.0, precision="Q4_K_M"):
    """Generate full configuration for N providers."""
    
    config = {
        "model": {
            "name": model_name,
            "total_layers": total_layers,
            "model_size_gb": model_size_gb,
            "precision": precision,
            "vocab_size": 32000,
            "hidden_dim": 2048
        },
        "cluster": {
            "num_providers": num_providers,
            "topology": "chain",  # Provider 1 -> Provider 2 -> ... -> Provider N
            "communication": "grpc",
            "fallback": "http"
        },
        "shards": calculate_shards(num_providers, total_layers, model_size_gb, precision),
        "deployment": {
            "docker_compose_file": "docker/docker-compose-15.yml",
            "provider_image": "dicai-provider:latest",
            "admin_image": "dicai-admin:latest"
        }
    }
    
    return config


def generate_docker_compose(config, output_file):
    """Generate docker-compose for N providers."""
    
    services = {
        "admin": {
            "build": {
                "context": "..",
                "dockerfile": "docker/Dockerfile.admin"
            },
            "ports": ["8080:8080"],
            "volumes": [
                "../models:/app/models:ro",
                "../pytorch_model:/app/pytorch_model"
            ],
            "networks": ["dicai-network"]
        }
    }
    
    for shard in config["shards"]:
        provider_id = shard["provider_id"]
        port = shard["port"]
        start_layer = shard["start_layer"]
        end_layer = shard["end_layer"]
        
        services[provider_id] = {
            "build": {
                "context": "..",
                "dockerfile": "docker/Dockerfile.provider"
            },
            "ports": [f"{port}:50051"],
            "volumes": [
                f"../pytorch_model/{provider_id}:/app/shard:ro"
            ],
            "environment": {
                "PROVIDER_ID": provider_id,
                "START_LAYER": str(start_layer),
                "END_LAYER": str(end_layer),
                "PORT": "50051"
            },
            "command": [
                "python3", "provider/grpc_provider.py",
                "--id", provider_id,
                "--shard-dir", "/app/shard",
                "--start-layer", str(start_layer),
                "--end-layer", str(end_layer),
                "--port", "50051"
            ],
            "networks": ["dicai-network"],
            "depends_on": ["admin"]
        }
    
    compose = {
        "version": "3.8",
        "services": services,
        "networks": {
            "dicai-network": {
                "driver": "bridge"
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(compose, f, indent=2)
    
    print(f"Generated docker-compose: {output_file}")


def generate_test_script(config, output_file):
    """Generate test script for N providers."""
    
    script = f'''#!/usr/bin/env python3
"""
DiCAI {config['cluster']['num_providers']}-Provider Test Script

Auto-generated test for {config['model']['name']} distributed across {config['cluster']['num_providers']} providers.
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
'''
    
    for shard in config["shards"]:
        script += f'    "localhost:{shard["port"]}",  # {shard["provider_id"]} (layers {shard["start_layer"]}-{shard["end_layer"]})\n'
    
    script += ''']


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
    print(f"\\n{{'='*60}}")
    print(f"Testing {len(PROVIDERS)}-Provider Chain")
    print(f"{{'='*60}}")
    
    # Start with input IDs
    input_ids = [1, 2, 3]
    hidden_states = None
    hidden_shape = None
    
    for i, address in enumerate(PROVIDERS):
        print(f"\\n[Provider {i+1}] {{address}}")
        
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
        
        print(f"\\n{{'='*60}}")
        print(f"SUCCESS! Predicted token: {{predicted_token}}")
        print(f"{{'='*60}}")
    else:
        print("\\nNo logits from final provider")


if __name__ == "__main__":
    test_end_to_end()
'''
    
    with open(output_file, 'w') as f:
        f.write(script)
    
    os.chmod(output_file, 0o755)
    print(f"Generated test script: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="DiCAI Provider Config Generator")
    parser.add_argument("--providers", type=int, default=15, help="Number of providers")
    parser.add_argument("--model", default="tinyllama-1.1b", help="Model name")
    parser.add_argument("--layers", type=int, default=22, help="Total layers")
    parser.add_argument("--size", type=float, default=4.0, help="Model size in GB")
    parser.add_argument("--precision", default="Q4_K_M", help="Precision")
    parser.add_argument("--output-dir", default="configs", help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate config
    config = generate_config(args.providers, args.model, args.layers, args.size, args.precision)
    
    config_file = os.path.join(args.output_dir, f"{args.providers}_providers.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\\nGenerated config: {config_file}")
    print(f"\\nModel: {args.model}")
    print(f"Total layers: {args.layers}")
    print(f"Providers: {args.providers}")
    print(f"Layers per provider: ~{math.ceil(args.layers / args.providers)}")
    print(f"Shard size: ~{config['shards'][0]['approx_size_gb']:.2f} GB each")
    
    # Generate docker-compose
    compose_file = os.path.join(args.output_dir, f"docker-compose-{args.providers}.yml")
    generate_docker_compose(config, compose_file)
    
    # Generate test script
    test_file = os.path.join(args.output_dir, f"test_{args.providers}_providers.py")
    generate_test_script(config, test_file)
    
    print(f"\\nFiles generated in {args.output_dir}/")
    print(f"  - {config_file}")
    print(f"  - {compose_file}")
    print(f"  - {test_file}")


if __name__ == "__main__":
    main()
