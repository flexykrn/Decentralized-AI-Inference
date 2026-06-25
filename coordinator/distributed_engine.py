#!/usr/bin/env python3
"""
DiCAI True Distributed Model Sharding

Uses llama.cpp's RPC backend for automatic distributed inference.
Each provider runs rpc-server (exposes compute device).
Coordinator runs llama-server with --rpc flag.

This is NOT mock - it uses real llama.cpp distributed execution.
Model layers are automatically distributed across RPC backends.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List, Optional
import threading


class DistributedInferenceEngine:
    """
    Real distributed inference using llama.cpp RPC backend.
    
    Architecture:
    - Providers: run rpc-server (each exposes a GPU/CPU device)
    - Coordinator: runs llama-server with --rpc flag
    - llama.cpp automatically splits model layers across providers
    """
    
    def __init__(self, model_path: str, port: int = 8080):
        self.model_path = model_path
        self.port = port
        self.providers: Dict[str, dict] = {}
        self.llama_server_proc: Optional[subprocess.Popen] = None
        self.rpc_procs: List[subprocess.Popen] = []
        self.running = False
        
    def find_binary(self, name: str) -> str:
        """Find llama.cpp binary."""
        paths = [
            f"/tmp/llama.cpp/build/bin/{name}",
            f"/usr/local/bin/{name}",
            f"/usr/bin/{name}",
        ]
        for path in paths:
            if os.path.exists(path):
                return path
        
        import shutil
        found = shutil.which(name)
        if found:
            return found
        
        raise FileNotFoundError(
            f"Binary '{name}' not found. Build llama.cpp with: "
            "cmake -B build -DGGML_RPC=ON && cmake --build build --target llama-server rpc-server"
        )
    
    def add_provider(self, provider_id: str, host: str, port: int, device_type: str = "cpu"):
        """Register an external provider running rpc-server."""
        self.providers[provider_id] = {
            "id": provider_id,
            "host": host,
            "port": port,
            "device_type": device_type,
            "status": "registered",
        }
        print(f"[Engine] Registered provider {provider_id} at {host}:{port} ({device_type})")
    
    def start_local_providers(self, count: int, base_port: int = 50052):
        """Start local RPC providers for testing."""
        rpc_server = self.find_binary("rpc-server")
        
        for i in range(count):
            port = base_port + i
            cmd = [rpc_server, "-H", "0.0.0.0", "-p", str(port), "-t", "4"]
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.rpc_procs.append(proc)
            self.add_provider(f"local-{i}", "localhost", port, "cpu")
            print(f"[Engine] Started local provider {i} on port {port} (PID: {proc.pid})")
        
        # Wait for RPC servers to initialize
        time.sleep(2)
        
        # Verify they're running
        for proc in self.rpc_procs:
            if proc.poll() is not None:
                print(f"[Engine] WARNING: RPC server (PID: {proc.pid}) exited early")
    
    def build_rpc_string(self) -> str:
        """Build comma-separated RPC backend string."""
        return ",".join([
            f"{p['host']}:{p['port']}"
            for p in self.providers.values()
        ])
    
    def start(self, ctx_size: int = 2048, verbose: bool = False) -> bool:
        """Start the distributed inference engine."""
        llama_server = self.find_binary("llama-server")
        rpc_string = self.build_rpc_string()
        
        if not rpc_string:
            print("[Engine] ERROR: No providers registered")
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
        
        print(f"[Engine] Starting distributed inference")
        print(f"[Engine] Model: {self.model_path}")
        print(f"[Engine] Providers: {len(self.providers)}")
        print(f"[Engine] RPC backends: {rpc_string}")
        print(f"[Engine] Port: {self.port}")
        
        self.llama_server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        
        # Wait for server to be ready
        print("[Engine] Waiting for llama-server to start...")
        for attempt in range(60):
            time.sleep(1)
            
            # Check if process died
            if self.llama_server_proc.poll() is not None:
                stdout, _ = self.llama_server_proc.communicate()
                print(f"[Engine] ERROR: llama-server exited with code {self.llama_server_proc.returncode}")
                print(f"[Engine] Output: {stdout.decode()}")
                return False
            
            try:
                req = urllib.request.Request(
                    f"http://localhost:{self.port}/health",
                    method="GET"
                )
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read())
                        print(f"[Engine] llama-server ready: {data}")
                        self.running = True
                        return True
            except Exception:
                pass
        
        print("[Engine] WARNING: llama-server may not be ready yet")
        return True
    
    def stream_logs(self):
        """Stream llama-server logs in real-time."""
        if not self.llama_server_proc or not self.llama_server_proc.stdout:
            return
        
        try:
            for line in iter(self.llama_server_proc.stdout.readline, b''):
                print(f"[llama-server] {line.decode().rstrip()}")
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"[Engine] Log streaming error: {e}")
    
    def test_inference(self, prompt: str = "Hello, I am", max_tokens: int = 20) -> Optional[dict]:
        """Test inference via the OpenAI-compatible API."""
        if not self.running:
            print("[Engine] ERROR: Engine not running")
            return None
        
        url = f"http://localhost:{self.port}/completion"
        data = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }).encode()
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                print(f"[Engine] Inference test successful")
                print(f"[Engine] Prompt: '{prompt}'")
                print(f"[Engine] Output: '{result.get('content', '')}'")
                return result
        except Exception as e:
            print(f"[Engine] Inference test failed: {e}")
            return None
    
    def stop(self):
        """Stop all processes."""
        print("[Engine] Stopping distributed inference engine...")
        self.running = False
        
        if self.llama_server_proc:
            self.llama_server_proc.terminate()
            try:
                self.llama_server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.llama_server_proc.kill()
            print("[Engine] llama-server stopped")
        
        for proc in self.rpc_procs:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        
        print(f"[Engine] {len(self.rpc_procs)} RPC providers stopped")


def main():
    parser = argparse.ArgumentParser(description="DiCAI Distributed Inference Engine")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--port", type=int, default=8080, help="Coordinator port")
    parser.add_argument("--local-providers", type=int, default=2, help="Number of local RPC providers")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--test", action="store_true", help="Run inference test after startup")
    args = parser.parse_args()
    
    engine = DistributedInferenceEngine(args.model, args.port)
    
    # Start local providers
    if args.local_providers > 0:
        engine.start_local_providers(args.local_providers)
    
    # Start coordinator
    if engine.start(args.ctx_size, args.verbose):
        print(f"\n{'='*60}")
        print(f"DiCAI Distributed Inference Engine Ready")
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"Providers: {len(engine.providers)}")
        print(f"OpenAI endpoint: http://localhost:{args.port}/v1/chat/completions")
        print(f"Legacy endpoint: http://localhost:{args.port}/completion")
        print(f"{'='*60}\n")
        
        # Test inference
        if args.test:
            time.sleep(2)  # Give server time to fully initialize
            engine.test_inference()
        
        # Stream logs until interrupted
        print("Press Ctrl+C to stop\n")
        try:
            engine.stream_logs()
        except KeyboardInterrupt:
            pass
    
    engine.stop()


if __name__ == "__main__":
    main()
