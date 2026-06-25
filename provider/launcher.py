#!/usr/bin/env python3
"""
Provider Node Launcher
Starts llama-server and registers with the coordinator.
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request
import json


def get_llama_server_path():
    """Find llama-server binary."""
    paths = [
        "/tmp/llama.cpp/build/bin/llama-server",
        "/usr/local/bin/llama-server",
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    
    # Try which
    import shutil
    return shutil.which("llama-server") or "llama-server"


def register_with_coordinator(coordinator_url, provider_id, host, port):
    """Register this provider with the coordinator."""
    try:
        data = json.dumps({
            "id": provider_id,
            "host": host,
            "port": port,
        }).encode()
        
        req = urllib.request.Request(
            f"{coordinator_url}/api/v1/providers/register",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            print(f"[Provider] Registered: {result}")
            return True
    except Exception as e:
        print(f"[Provider] Registration failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="DiCAI Provider Node")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--port", type=int, default=8081, help="Port for llama-server")
    parser.add_argument("--coordinator", default="http://localhost:8080", help="Coordinator URL")
    parser.add_argument("--id", default="", help="Provider ID (auto-generated if empty)")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    args = parser.parse_args()
    
    provider_id = args.id or f"provider-{args.port}"
    
    llama_server = get_llama_server_path()
    print(f"[Provider] Using llama-server: {llama_server}")
    
    # Start llama-server
    cmd = [
        llama_server,
        "--model", args.model,
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--ctx-size", str(args.ctx_size),
        "--threads", str(args.threads),
    ]
    
    print(f"[Provider] Starting: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    
    # Wait for server to start
    print(f"[Provider] Waiting for llama-server to start...")
    time.sleep(3)
    
    # Check if server is running
    try:
        with urllib.request.urlopen(f"http://localhost:{args.port}/health", timeout=5) as resp:
            if resp.status == 200:
                print(f"[Provider] llama-server is running on port {args.port}")
    except Exception as e:
        print(f"[Provider] Server not responding yet, continuing anyway...")
    
    # Register with coordinator
    print(f"[Provider] Registering with coordinator at {args.coordinator}")
    for attempt in range(5):
        if register_with_coordinator(args.coordinator, provider_id, "localhost", args.port):
            break
        time.sleep(2)
    else:
        print("[Provider] Failed to register with coordinator")
    
    print(f"[Provider] Ready. Press Ctrl+C to stop.")
    
    try:
        # Stream logs
        for line in iter(process.stdout.readline, b''):
            print(f"[llama-server] {line.decode().rstrip()}")
    except KeyboardInterrupt:
        print("\n[Provider] Shutting down...")
    finally:
        process.terminate()
        process.wait()


if __name__ == "__main__":
    main()
