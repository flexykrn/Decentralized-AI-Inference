import argparse
import requests
import time
import json
import subprocess
import sys
import os

from provider.detect import detect_hardware
from shared.grpc_server import start_tensor_server

# Global state
model_process = None
tensor_server = None

def download_model_shard(admin_url, model_id, shard_id, device_id):
    """Download assigned model shard from admin."""
    print(f"[Provider {device_id}] Downloading shard {shard_id} for model {model_id}...")
    
    # Download from admin's shard endpoint
    url = f"{admin_url}/api/v3/models/{model_id}/shards/{shard_id}/download"
    resp = requests.get(url, stream=True)
    
    if resp.status_code != 200:
        print(f"[Provider {device_id}] Download failed: {resp.text}")
        return False
    
    # Save to local models directory
    os.makedirs(f"models/{model_id}", exist_ok=True)
    shard_path = f"models/{model_id}/{shard_id}.gguf"
    
    with open(shard_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"[Provider {device_id}] Shard downloaded to {shard_path}")
    return True

def launch_exo_worker(model_id, layer_start, layer_end, port, device_id):
    """Launch exo worker process for assigned layers."""
    global model_process
    
    print(f"[Provider {device_id}] Launching exo worker for layers {layer_start}-{layer_end}...")
    
    # Check if exo is installed
    try:
        subprocess.run(["exo", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"[Provider {device_id}] exo not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "exo"], check=True)
    
    # Launch exo worker
    cmd = [
        "exo", "worker",
        "--model", model_id,
        "--layers", f"{layer_start}-{layer_end}",
        "--port", str(port),
        "--device-id", device_id,
    ]
    
    model_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    print(f"[Provider {device_id}] exo worker started (PID: {model_process.pid})")
    return model_process

def launch_llama_cpp_server(model_path, port, device_id):
    """Fallback: Launch llama.cpp server for model shard."""
    global model_process
    
    print(f"[Provider {device_id}] Launching llama.cpp server...")
    
    cmd = [
        "python3", "-m", "llama_cpp.server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    
    model_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    print(f"[Provider {device_id}] llama.cpp server started (PID: {model_process.pid})")
    return model_process

def main():
    parser = argparse.ArgumentParser(description="DiCAI Provider Client")
    parser.add_argument("--admin", default="http://localhost:8080", help="Admin URL")
    parser.add_argument("--device-id", default="auto", help="Device ID (auto=detect)")
    parser.add_argument("--token", required=True, help="Invite token from admin")
    parser.add_argument("--grpc-port", type=int, default=50051, help="gRPC port for tensor passing")
    args = parser.parse_args()
    
    # Detect hardware
    hw = detect_hardware()
    device_id = args.device_id if args.device_id != "auto" else hw["device_id"]
    
    print(f"[Provider {device_id}] Detected hardware:")
    print(json.dumps(hw, indent=2))
    
    # Register with admin
    resp = requests.post(
        f"{args.admin}/api/v3/providers/register",
        json={
            "device_id": device_id,
            "device_memory": hw.get("device_memory", hw.get("ram_gb", 16)),
            "memory_type": hw.get("memory_type", "ram"),
            "compute_backend": hw.get("compute_backend", "cpu"),
            "os_type": hw.get("os_type", "linux"),
            "gpu_name": hw.get("gpu_name"),
            "cpu_cores": hw.get("cpu_cores"),
            "invite_token": args.token,
        },
    )
    
    if resp.status_code != 200:
        print(f"[Provider {device_id}] Registration failed: {resp.text}")
        return
    
    data = resp.json()
    print(f"[Provider {device_id}] Registered. Matched models: {data.get('matched_models', [])}")
    
    # Start gRPC server for tensor passing
    global tensor_server
    tensor_server = start_tensor_server(device_id, args.grpc_port)
    print(f"[Provider {device_id}] gRPC tensor server started on port {args.grpc_port}")
    
    # Heartbeat loop
    print(f"[Provider {device_id}] Starting heartbeat loop...")
    while True:
        try:
            # Send heartbeat
            requests.post(f"{args.admin}/api/v3/providers/{device_id}/heartbeat", timeout=5)
            
            # Check for model assignments
            status_resp = requests.get(f"{args.admin}/api/v3/providers/{device_id}")
            if status_resp.status_code == 200:
                status = status_resp.json()
                # If assigned to a model and not yet loaded, download and launch
                if status.get("assigned_model") and not model_process:
                    model_id = status["assigned_model"]
                    layer_start = status.get("layer_start", 0)
                    layer_end = status.get("layer_end", 0)
                    
                    print(f"[Provider {device_id}] Assigned to model {model_id} layers {layer_start}-{layer_end}")
                    
                    # Download shard
                    shard_id = f"{model_id}_{layer_start}_{layer_end}"
                    if download_model_shard(args.admin, model_id, shard_id, device_id):
                        # Launch worker
                        shard_path = f"models/{model_id}/{shard_id}.gguf"
                        launch_exo_worker(model_id, layer_start, layer_end, args.grpc_port, device_id)
            
        except Exception as e:
            print(f"[Provider {device_id}] Heartbeat/assignment check failed: {e}")
        
        time.sleep(5)

if __name__ == "__main__":
    main()
