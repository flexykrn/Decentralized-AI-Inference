#!/usr/bin/env python3
"""
DiCAI - Provider Daemon

Each provider:
1. Detects hardware capabilities
2. Registers with DHT coordinator
3. Downloads assigned layers from admin or peers
4. Serves /forward endpoint for inference
5. Heartbeats every 5 seconds

Usage:
    python -m dicai.provider --id p1 --coordinator http://admin:8464 --layer-server http://admin:9000 --port 5001
"""

import os
import sys
import time
import json
import argparse
import threading
import psutil
import requests
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dht.discovery import DHTClient, ProviderInfo
from src.admin.layer_server import LayerDownloader, PeerDiscovery


class HardwareDetector:
    """Detects local hardware capabilities."""
    
    @staticmethod
    def detect() -> Dict[str, Any]:
        """Detect hardware and return capability report."""
        info = {
            "cpu_cores": psutil.cpu_count(logical=True),
            "cpu_freq_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else 0,
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
            "gpu_available": False,
            "gpu_memory_gb": 0.0,
            "disk_free_gb": psutil.disk_usage('/').free / (1024**3),
            "internet_mbps": 0.0,  # TODO: speed test
        }
        
        # Try to detect GPU
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu_available"] = True
                info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            elif torch.backends.mps.is_available():
                info["gpu_available"] = True
                info["gpu_memory_gb"] = psutil.virtual_memory().total / (1024**3) * 0.5  # MPS uses unified memory
        except ImportError:
            pass
            
        return info
    
    @staticmethod
    def calculate_capacity_score(info: Dict[str, Any]) -> float:
        """Calculate capacity score for layer assignment."""
        return (
            info["ram_gb"] * 0.6 +
            info["cpu_cores"] * 0.2 +
            info["gpu_memory_gb"] * 0.15 +
            info["internet_mbps"] * 0.05
        )


class ProviderService:
    """Provider daemon that serves inference requests."""
    
    def __init__(self, provider_id: str, coordinator_url: str, layer_server_url: str,
                 port: int = 5001, host: str = "0.0.0.0", invite_code: str = ""):
        self.provider_id = provider_id
        self.coordinator_url = coordinator_url.rstrip("/")
        self.layer_server_url = layer_server_url.rstrip("/")
        self.port = port
        self.host = host
        self.invite_code = invite_code or os.environ.get("DICAI_PROVIDER_CODE", "")

        self.hardware = HardwareDetector.detect()
        self.capacity_score = HardwareDetector.calculate_capacity_score(self.hardware)

        self.dht_client = DHTClient(
            coordinator_url.replace("http://", "").replace("https://", "").split(":")[0],
            int(coordinator_url.split(":")[-1].split("/")[0]) if ":" in coordinator_url else 8464
        )

        self.layer_downloader = LayerDownloader(layer_server_url, cache_dir=f"cache/{provider_id}")
        self.peer_discovery = PeerDiscovery(self.dht_client)
        
        self.assigned_layers = (0, 0)  # (start, end)
        self.model_path: str = ""
        self.model = None
        
        self.app = FastAPI(title=f"DiCAI Provider {provider_id}")
        self._setup_routes()
        
        self.running = False
        self.heartbeat_thread = None
        
    def _setup_routes(self):
        """Setup HTTP endpoints."""
        
        @self.app.post("/forward")
        async def forward(request: Dict[str, Any]):
            """Execute forward pass on assigned layers."""
            return await self._handle_forward(request)
            
        @self.app.get("/health")
        async def health():
            """Health check."""
            return {
                "provider_id": self.provider_id,
                "status": "healthy" if self.running else "offline",
                "assigned_layers": list(self.assigned_layers),
                "hardware": self.hardware,
                "memory_used_gb": psutil.virtual_memory().used / (1024**3),
                "memory_total_gb": self.hardware.get("ram_gb", 0),
                "model_loaded": self.model is not None
            }

        @self.app.get("/ready")
        async def ready():
            """Readiness probe: true only when model is loaded and provider is running."""
            return {
                "provider_id": self.provider_id,
                "ready": self.running and self.model is not None,
                "model_loaded": self.model is not None,
                "assigned_layers": list(self.assigned_layers)
            }

        @self.app.get("/status")
        async def status():
            """Detailed bootstrap status."""
            return {
                "provider_id": self.provider_id,
                "running": self.running,
                "model_loaded": self.model is not None,
                "assigned_layers": list(self.assigned_layers),
                "model_path": self.model_path
            }

        @self.app.post("/reload")
        async def reload(request: Dict[str, Any]):
            """Reload with new layer assignment."""
            start_layer = request.get("start_layer", 0)
            end_layer = request.get("end_layer", 0)
            await self._reload_layers(start_layer, end_layer)
            return {"status": "reloaded", "layers": (start_layer, end_layer)}
            
    async def _handle_forward(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle inference forward pass."""
        # For MVP: we use llama.cpp to generate tokens
        # The provider loads the full model but only "claims" certain layers
        # In reality, llama.cpp handles all layers internally
        # Future: implement true layer-by-layer forward with PyTorch
        
        input_ids = request.get("input_ids", [])
        next_provider = request.get("next_provider")
        
        if not self.model:
            return {"error": "Model not loaded"}
            
        # Generate next token
        # For MVP, we generate all tokens here and return
        # In distributed mode, we'd only compute our assigned layers and forward
        
        try:
            # llama.cpp expects a prompt string, not input_ids
            prompt = request.get("prompt", "")
            if not prompt and input_ids:
                prompt = " "
                
            response = self.model(
                prompt,
                max_tokens=request.get("max_tokens", 20),
                temperature=request.get("temperature", 0.7),
                stop=["</s>"],
            )
            
            # Handle llama.cpp response format
            text = ""
            if isinstance(response, dict) and 'choices' in response:
                text = response['choices'][0].get('text', '')
            
            return {
                "text": text,
                "token": text.split()[0] if text.split() else "",
                "provider_id": self.provider_id,
                "layers_processed": self.assigned_layers,
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def _reload_layers(self, start_layer: int, end_layer: int):
        """Reload with new layer assignment."""
        print(f"[Provider {self.provider_id}] Reloading layers {start_layer}-{end_layer}")
        self.assigned_layers = (start_layer, end_layer)
        
        # Download new layers if needed
        layer_indices = list(range(start_layer, end_layer + 1))
        
        # Find peers that have these layers
        peers = self.peer_discovery.find_peers_with_layers(layer_indices)
        
        # Download from peers or admin
        successful = self.layer_downloader.download_layers(layer_indices, peer_urls=peers)
        print(f"[Provider {self.provider_id}] Downloaded {len(successful)}/{len(layer_indices)} layers")
        
        # Update DHT registration
        self._register_with_coordinator()
        
    def _register_with_coordinator(self):
        """Register with DHT coordinator."""
        # Re-request assignment after model is loaded to ensure it reflects current topology
        new_assignment = self._request_assignment()
        if new_assignment != (0, 0):
            self.assigned_layers = new_assignment
            print(f"[Provider {self.provider_id}] Final assignment: layers {self.assigned_layers[0]}-{self.assigned_layers[1]}")

        provider = ProviderInfo(
            provider_id=self.provider_id,
            address=self.host if self.host != "0.0.0.0" else "127.0.0.1",
            port=self.port,
            start_layer=self.assigned_layers[0],
            end_layer=self.assigned_layers[1],
            total_layers=22,  # TODO: get from manifest
            last_seen=time.time(),
            status="online",
            throughput=self.hardware["cpu_cores"] * 10.0,  # Rough estimate
            gpu_available=self.hardware["gpu_available"],
            gpu_memory_gb=self.hardware["gpu_memory_gb"],
            cpu_memory_gb=self.hardware["ram_gb"],
        )
        
        try:
            # Use HTTP coordinator registration with invite code
            resp = requests.post(
                f"{self.coordinator_url}/register",
                json={"code": self.invite_code, "provider": provider.to_dict()},
                timeout=10
            )
            if resp.status_code == 200:
                result = resp.json()
                print(f"[Provider {self.provider_id}] Registered with coordinator: {result}")
            else:
                print(f"[Provider {self.provider_id}] Registration failed: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"[Provider {self.provider_id}] Failed to register: {e}")
            
    def _heartbeat_loop(self):
        """Send periodic heartbeats to coordinator."""
        while self.running:
            try:
                self.dht_client.heartbeat(self.provider_id)
            except Exception as e:
                print(f"[Provider {self.provider_id}] Heartbeat failed: {e}")
                
            time.sleep(5)
            
    def start(self):
        """Start the provider service."""
        print(f"[Provider {self.provider_id}] Starting...")
        print(f"[Provider {self.provider_id}] Hardware: {json.dumps(self.hardware, indent=2)}")
        print(f"[Provider {self.provider_id}] Capacity score: {self.capacity_score:.2f}")

        # Start HTTP server immediately so the provider is always reachable
        self.running = True
        print(f"[Provider {self.provider_id}] Serving on {self.host}:{self.port}")
        threading.Thread(target=self._bootstrap_loop, daemon=True).start()
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")
        return True

    def _bootstrap_loop(self):
        """Background: download layers, reassemble model, register."""
        try:
            self._do_bootstrap()
        except Exception as e:
            print(f"[Provider {self.provider_id}] Bootstrap crash: {e}")
            import traceback
            traceback.print_exc()

    def _do_bootstrap(self):
        """Download layers, reassemble model, register."""
        # Download manifest with retry
        manifest = None
        for attempt in range(10):
            manifest = self.layer_downloader.download_manifest()
            if manifest:
                break
            print(f"[Provider {self.provider_id}] Manifest download attempt {attempt+1} failed, retrying...")
            time.sleep(2)

        if not manifest:
            print(f"[Provider {self.provider_id}] Failed to download manifest after retries")
            self.running = False
            return

        print(f"[Provider {self.provider_id}] Manifest downloaded: {manifest['model_name']}")

        # Request layer assignment from coordinator with retry
        self.assigned_layers = self._request_assignment()
        if self.assigned_layers == (0, 0):
            print(f"[Provider {self.provider_id}] Using fallback full assignment")
            self.assigned_layers = (0, manifest.get('n_layers', 22) - 1)

        # Download assigned layers
        layer_indices = list(range(self.assigned_layers[0], self.assigned_layers[1] + 1))
        peers = self.peer_discovery.find_peers_with_layers(layer_indices)
        successful = self.layer_downloader.download_layers(layer_indices, peer_urls=peers)
        print(f"[Provider {self.provider_id}] Downloaded {len(successful)}/{len(layer_indices)} layers")

        # Reassemble full GGUF from chunks if layer format is gguf_chunks
        if manifest.get('format') == 'gguf_chunks':
            reassembled_path = self._reassemble_gguf(manifest, self.layer_downloader.cache_dir)
            if reassembled_path:
                self.model_path = reassembled_path

        # Load model
        if not self.model_path:
            self.model_path = manifest.get("original_path", "")

        if os.path.exists(self.model_path):
            try:
                import llama_cpp
                self.model = llama_cpp.Llama(
                    model_path=self.model_path,
                    n_ctx=512,
                    verbose=False
                )
                print(f"[Provider {self.provider_id}] Model loaded: {self.model_path}")
            except Exception as e:
                print(f"[Provider {self.provider_id}] Failed to load model: {e}")
                self.model = None
        else:
            print(f"[Provider {self.provider_id}] Model not found at {self.model_path}")

        # Register with coordinator
        self._register_with_coordinator()

        # Start heartbeat
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

    def _request_assignment(self):
        """Ask coordinator for layer assignment."""
        try:
            url = f"{self.coordinator_url}/assign/{self.provider_id}"
            resp = requests.post(url, json={"capacity_score": self.capacity_score}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return (data.get("start_layer", 0), data.get("end_layer", 0))
        except Exception as e:
            print(f"[Provider {self.provider_id}] Assignment request failed: {e}")
        return (0, 0)

    def _reassemble_gguf(self, manifest: Dict[str, Any], cache_dir: str) -> Optional[str]:
        """Reassemble the full GGUF from downloaded chunks."""
        output_path = os.path.join(cache_dir, "model.gguf")
        if os.path.exists(output_path) and os.path.getsize(output_path) == manifest.get("total_size", 0):
            print(f"[Provider {self.provider_id}] Reassembled GGUF already exists: {output_path}")
            return output_path

        try:
            n_layers = manifest.get("n_layers", 22)
            with open(output_path, 'wb') as out:
                for idx in range(n_layers):
                    layer_file = os.path.join(cache_dir, f"layer_{idx:03d}.pt")
                    if not os.path.exists(layer_file):
                        raise FileNotFoundError(f"Missing layer file: {layer_file}")
                    data = torch.load(layer_file, map_location='cpu', weights_only=False)
                    out.write(data['chunk'])
            print(f"[Provider {self.provider_id}] Reassembled GGUF: {output_path}")
            return output_path
        except Exception as e:
            print(f"[Provider {self.provider_id}] Failed to reassemble GGUF: {e}")
            return None
        
    def stop(self):
        """Stop the provider service."""
        self.running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description="DiCAI Provider Daemon")
    parser.add_argument("--id", required=True, help="Provider ID")
    parser.add_argument("--coordinator", default="http://localhost:8464", help="Coordinator URL")
    parser.add_argument("--layer-server", default="http://localhost:9000", help="Layer server URL")
    parser.add_argument("--port", type=int, default=5001, help="Provider port")
    parser.add_argument("--host", default="0.0.0.0", help="Provider host")
    parser.add_argument("--code", default="", help="Invite code for coordinator registration")

    args = parser.parse_args()

    service = ProviderService(
        provider_id=args.id,
        coordinator_url=args.coordinator,
        layer_server_url=args.layer_server,
        port=args.port,
        host=args.host,
        invite_code=args.code
    )
    
    try:
        service.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        service.stop()


if __name__ == "__main__":
    main()
