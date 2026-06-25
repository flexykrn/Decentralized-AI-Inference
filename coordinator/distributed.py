#!/usr/bin/env python3
"""
DiCAI Distributed Inference Coordinator

Uses llama.cpp's RPC backend for automatic model distribution.
Each provider runs rpc-server, coordinator runs llama-server --rpc.

Architecture:
- Providers: run rpc-server (exposes compute device)
- Coordinator: runs llama-server with --rpc flag pointing to all providers
- Model is automatically distributed across RPC backends
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List


class DistributedCoordinator:
    def __init__(self, model_path: str, port: int = 8080):
        self.model_path = model_path
        self.port = port
        self.providers: Dict[str, dict] = {}
        self.llama_server_proc = None
        self.rpc_procs = []
        
    def find_binary(self, name: str) -> str:
        """Find llama.cpp binary."""
        paths = [
            f"/tmp/llama.cpp/build/bin/{name}",
            f"/usr/local/bin/{name}",
        ]
        for path in paths:
            if os.path.exists(path):
                return path
        
        import shutil
        found = shutil.which(name)
        if found:
            return found
        
        raise FileNotFoundError(f"Binary '{name}' not found. Build llama.cpp first.")
    
    def register_provider(self, provider_id: str, host: str, port: int):
        """Register a provider running rpc-server."""
        self.providers[provider_id] = {
            "id": provider_id,
            "host": host,
            "port": port,
            "status": "registered",
        }
        print(f"[Coordinator] Registered provider {provider_id} at {host}:{port}")
    
    def build_rpc_string(self) -> str:
        """Build comma-separated RPC backend string for llama-server."""
        return ",".join([
            f"{p['host']}:{p['port']}"
            for p in self.providers.values()
        ])
    
    def start_local_rpc(self, num_providers: int, base_port: int = 50052):
        """Start local RPC servers for testing."""
        rpc_server = self.find_binary("rpc-server")
        
        for i in range(num_providers):
            port = base_port + i
            cmd = [rpc_server, "-H", "0.0.0.0", "-p", str(port), "-t", "4"]
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.rpc_procs.append(proc)
            self.register_provider(f"local-{i}", "localhost", port)
            print(f"[Coordinator] Started local RPC server {i} on port {port} (PID: {proc.pid})")
        
        # Wait for RPC servers to start
        time.sleep(2)
    
    def start(self, ctx_size: int = 2048, verbose: bool = False):
        """Start the coordinator (llama-server with RPC backends)."""
        llama_server = self.find_binary("llama-server")
        rpc_string = self.build_rpc_string()
        
        if not rpc_string:
            print("[Coordinator] ERROR: No providers registered")
            return False
        
        cmd = [
            llama_server,
            "--model", self.model_path,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--ctx-size", str(ctx_size),
            "--rpc", rpc_string,
        ]
        
        if verbose:
            cmd.append("--verbose")
        
        print(f"[Coordinator] Starting llama-server with RPC backends: {rpc_string}")
        print(f"[Coordinator] Command: {' '.join(cmd)}")
        
        self.llama_server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        
        # Wait for server to start
        print("[Coordinator] Waiting for llama-server to start...")
        for attempt in range(30):
            time.sleep(1)
            try:
                req = urllib.request.Request(
                    f"http://localhost:{self.port}/health",
                    method="GET"
                )
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        print(f"[Coordinator] llama-server ready on port {self.port}")
                        return True
            except Exception:
                pass
        
        print("[Coordinator] WARNING: llama-server may not be ready yet")
        return True
    
    def stream_logs(self):
        """Stream llama-server logs."""
        if not self.llama_server_proc:
            return
        
        try:
            for line in iter(self.llama_server_proc.stdout.readline, b''):
                print(f"[llama-server] {line.decode().rstrip()}")
        except KeyboardInterrupt:
            pass
    
    def stop(self):
        """Stop all processes."""
        print("[Coordinator] Stopping...")
        
        if self.llama_server_proc:
            self.llama_server_proc.terminate()
            self.llama_server_proc.wait()
            print("[Coordinator] llama-server stopped")
        
        for proc in self.rpc_procs:
            proc.terminate()
            proc.wait()
        
        print(f"[Coordinator] {len(self.rpc_procs)} RPC servers stopped")


def main():
    parser = argparse.ArgumentParser(description="DiCAI Distributed Coordinator")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--port", type=int, default=8080, help="Coordinator port")
    parser.add_argument("--local-providers", type=int, default=2, help="Number of local RPC providers to start")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    coordinator = DistributedCoordinator(args.model, args.port)
    
    # Start local RPC providers
    coordinator.start_local_rpc(args.local_providers)
    
    # Start coordinator
    if coordinator.start(args.ctx_size, args.verbose):
        print(f"\n{'='*60}")
        print(f"DiCAI Distributed Inference Ready")
        print(f"{'='*60}")
        print(f"OpenAI endpoint: http://localhost:{args.port}/v1/chat/completions")
        print(f"Providers: {args.local_providers}")
        print(f"Press Ctrl+C to stop")
        print(f"{'='*60}\n")
        
        coordinator.stream_logs()
    
    coordinator.stop()


if __name__ == "__main__":
    main()
