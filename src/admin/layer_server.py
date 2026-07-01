#!/usr/bin/env python3
"""
DiCAI - Layer Distribution Server (BitTorrent-style)

Admin seeds the full model. Providers download only assigned layers.
New providers get layers from nearest peers, not admin.

Architecture:
- Admin runs HTTP seed server with all layer files
- Providers have HTTP server too (peer-to-peer)
- New provider asks DHT for peers with needed layers
- Downloads from nearest peer, falls back to admin
- After download, announces to DHT: "I have layers X-Y"

Usage:
    # Admin
    python -m dicai.admin.layer_server --port 9000 --layers-dir layers/

    # Provider (downloads from admin + peers)
    python -m dicai.provider --layer-server http://admin:9000
"""

import os
import sys
import json
import hashlib
import time
import threading
import requests
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dht.discovery import DHTClient, ProviderInfo


@dataclass
class LayerFile:
    """Information about a layer file."""
    index: int
    file_path: str
    checksum: str
    size_bytes: int


class LayerServer:
    """HTTP server for serving layer files. Run by admin and by providers (as peers)."""
    
    def __init__(self, layers_dir: str, port: int = 9000, host: str = "0.0.0.0"):
        self.layers_dir = layers_dir
        self.port = port
        self.host = host
        self.app = FastAPI(title="DiCAI Layer Server")
        self.manifest: Optional[dict] = None
        self.available_layers: Set[int] = set()
        
        self._load_manifest()
        self._setup_routes()
        
    def _load_manifest(self):
        """Load manifest.json from layers directory."""
        manifest_path = os.path.join(self.layers_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
            # Track which layers we have locally
            for layer_info in self.manifest.get("layers", []):
                layer_file = os.path.join(self.layers_dir, layer_info["file"])
                if os.path.exists(layer_file):
                    self.available_layers.add(layer_info["index"])
            print(f"[LayerServer] Loaded manifest: {len(self.available_layers)} layers available")
        else:
            print(f"[LayerServer] No manifest found in {self.layers_dir}")
            
    def _setup_routes(self):
        """Setup HTTP routes."""
        
        @self.app.get("/manifest")
        async def get_manifest():
            """Get layer manifest."""
            if not self.manifest:
                raise HTTPException(status_code=404, detail="No manifest available")
            return self.manifest
            
        @self.app.get("/layer/{layer_index}")
        async def get_layer(layer_index: int):
            """Download a specific layer file."""
            if layer_index not in self.available_layers:
                raise HTTPException(status_code=404, detail=f"Layer {layer_index} not available")
                
            # Find the file for this layer
            layer_file = None
            for layer_info in self.manifest.get("layers", []):
                if layer_info["index"] == layer_index:
                    layer_file = os.path.join(self.layers_dir, layer_info["file"])
                    break
                    
            if not layer_file or not os.path.exists(layer_file):
                raise HTTPException(status_code=404, detail=f"Layer file not found")
                
            return FileResponse(layer_file, filename=f"layer_{layer_index:03d}.pt")
            
        @self.app.get("/layers")
        async def list_layers():
            """List available layers."""
            return {
                "available_layers": sorted(list(self.available_layers)),
                "count": len(self.available_layers),
                "model_name": self.manifest.get("model_name") if self.manifest else None
            }
            
        @self.app.get("/health")
        async def health():
            """Health check."""
            return {
                "status": "healthy",
                "available_layers": len(self.available_layers),
                "port": self.port
            }
            
    def run(self):
        """Run the server."""
        print(f"[LayerServer] Starting on {self.host}:{self.port}")
        print(f"[LayerServer] Serving layers from: {self.layers_dir}")
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")


class LayerDownloader:
    """Downloads layer files from admin or peers."""
    
    def __init__(self, admin_url: str, cache_dir: str = "layer_cache"):
        self.admin_url = admin_url.rstrip("/")
        self.cache_dir = cache_dir
        self.local_layers: Set[int] = set()
        self.layer_checksums: Dict[int, str] = {}
        
        os.makedirs(cache_dir, exist_ok=True)
        
    def download_manifest(self) -> Optional[dict]:
        """Download manifest from admin."""
        try:
            response = requests.get(f"{self.admin_url}/manifest", timeout=10)
            if response.status_code == 200:
                manifest = response.json()
                # Cache checksums
                for layer_info in manifest.get("layers", []):
                    self.layer_checksums[layer_info["index"]] = layer_info["checksum"]
                return manifest
        except Exception as e:
            print(f"[Downloader] Failed to download manifest: {e}")
        return None
        
    def download_layer(self, layer_index: int, peer_urls: Optional[List[str]] = None) -> bool:
        """Download a layer file from admin or peers."""
        # Check if already cached
        cached_file = os.path.join(self.cache_dir, f"layer_{layer_index:03d}.pt")
        if os.path.exists(cached_file):
            # Verify checksum
            expected_checksum = self.layer_checksums.get(layer_index)
            if expected_checksum:
                with open(cached_file, 'rb') as f:
                    actual_checksum = hashlib.sha256(f.read()).hexdigest()[:16]
                if actual_checksum == expected_checksum:
                    print(f"[Downloader] Layer {layer_index} already cached")
                    self.local_layers.add(layer_index)
                    return True
                else:
                    print(f"[Downloader] Layer {layer_index} checksum mismatch, re-downloading")
                    
        # Try peers first, then admin
        sources: List[str] = (peer_urls or []) + [self.admin_url]
        
        for source in sources:
            try:
                url = f"{source}/layer/{layer_index}"
                print(f"[Downloader] Downloading layer {layer_index} from {source}...")
                response = requests.get(url, timeout=30, stream=True)
                
                if response.status_code == 200:
                    with open(cached_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            
                    # Verify checksum
                    expected_checksum = self.layer_checksums.get(layer_index)
                    if expected_checksum:
                        with open(cached_file, 'rb') as f:
                            actual_checksum = hashlib.sha256(f.read()).hexdigest()[:16]
                        if actual_checksum != expected_checksum:
                            print(f"[Downloader] Checksum mismatch for layer {layer_index}")
                            os.remove(cached_file)
                            continue
                            
                    self.local_layers.add(layer_index)
                    print(f"[Downloader] Layer {layer_index} downloaded successfully")
                    return True
                    
            except Exception as e:
                print(f"[Downloader] Failed to download from {source}: {e}")
                continue
                
        print(f"[Downloader] Could not download layer {layer_index} from any source")
        return False
        
    def download_layers(self, layer_indices: List[int], peer_urls: Optional[List[str]] = None, max_workers: int = 2) -> List[int]:
        """Download multiple layers with limited concurrency."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        successful = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_layer = {}
            for idx in layer_indices:
                future = executor.submit(self.download_layer, idx, peer_urls)
                future_to_layer[future] = idx

            for future in as_completed(future_to_layer):
                idx = future_to_layer[future]
                try:
                    if future.result():
                        successful.append(idx)
                except Exception as e:
                    print(f"[Downloader] Layer {idx} download error: {e}")

        return successful
        
    def get_local_path(self, layer_index: int) -> Optional[str]:
        """Get local path for a cached layer."""
        if layer_index not in self.local_layers:
            return None
        return os.path.join(self.cache_dir, f"layer_{layer_index:03d}.pt")
        
    def has_layer(self, layer_index: int) -> bool:
        """Check if layer is available locally."""
        return layer_index in self.local_layers


class PeerDiscovery:
    """Discovers peers that have specific layers."""
    
    def __init__(self, dht_client: DHTClient):
        self.dht_client = dht_client
        
    def find_peers_with_layers(self, layer_indices: List[int]) -> List[str]:
        """Find peers that have the requested layers."""
        # Query DHT for providers
        providers = self.dht_client.list_providers()
        
        peers = []
        for provider_data in providers:
            provider = ProviderInfo.from_dict(provider_data)
            # Check if provider covers any of the requested layers
            for idx in layer_indices:
                if provider.start_layer <= idx <= provider.end_layer:
                    peer_url = f"http://{provider.address}:{provider.port}"
                    if peer_url not in peers:
                        peers.append(peer_url)
                    break
                    
        return peers


def test_layer_server():
    """Test layer server."""
    print("=" * 60)
    print("Layer Server Test")
    print("=" * 60)
    
    # Create test layers directory
    os.makedirs("test_layers", exist_ok=True)
    
    # Create test manifest
    manifest = {
        "model_name": "test-model",
        "n_layers": 3,
        "layers": [
            {"index": 0, "file": "layer_000.pt", "checksum": "abc123", "size_mb": 1.0},
            {"index": 1, "file": "layer_001.pt", "checksum": "def456", "size_mb": 1.0},
            {"index": 2, "file": "layer_002.pt", "checksum": "ghi789", "size_mb": 1.0},
        ]
    }
    
    with open("test_layers/manifest.json", 'w') as f:
        json.dump(manifest, f)
        
    # Create dummy layer files
    for i in range(3):
        with open(f"test_layers/layer_{i:03d}.pt", 'w') as f:
            f.write(f"dummy layer {i}")
            
    # Start server
    server = LayerServer("test_layers", port=9001)
    
    # Test manifest download
    downloader = LayerDownloader("http://localhost:9001")
    manifest = downloader.download_manifest()
    assert manifest is not None, "Failed to download manifest"
    print(f"Manifest downloaded: {manifest['model_name']}")
    
    # Test layer download
    success = downloader.download_layer(0)
    assert success, "Failed to download layer 0"
    print(f"Layer 0 downloaded")
    
    # Test from admin
    success = downloader.download_layer(1)
    assert success, "Failed to download layer 1"
    print(f"Layer 1 downloaded")
    
    print("\nLayer Server Test PASSED")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--layers-dir", default="layers")
    args = parser.parse_args()
    server = LayerServer(args.layers_dir, port=args.port, host=args.host)
    server.run()
