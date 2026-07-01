#!/usr/bin/env python3
"""Consolidated admin service: coordinator + layer server + API + web UI."""
import os
import sys
import time
import threading
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coordinator.main import Coordinator
from src.admin.layer_server import LayerServer
from src.api.server import DiCAIAPIServer


class AdminService:
    """Single process admin orchestrator."""

    def __init__(self, layers_dir: str = "layers", coordinator_port: int = 8464,
                 layer_port: int = 9000, api_port: int = 8080, host: str = "0.0.0.0"):
        self.layers_dir = layers_dir
        self.coordinator_port = coordinator_port
        self.layer_port = layer_port
        self.api_port = api_port
        self.host = host

        self.coordinator = Coordinator(port=coordinator_port, host=host)
        self.layer_server = LayerServer(layers_dir, port=layer_port, host=host)
        self.api_server = DiCAIAPIServer(
            coordinator_url=f"http://localhost:{coordinator_port}",
            port=api_port,
            host=host
        )

        # Mount static UI on API server
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
        self.api_server.app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @self.api_server.app.get("/")
        async def root():
            return FileResponse(os.path.join(static_dir, "index.html"))

        # Expose coordinator status via API
        @self.api_server.app.get("/status")
        async def status():
            route = self.coordinator._build_route()
            from dataclasses import asdict
            return {
                "status": "healthy",
                "providers": len(self.coordinator.dht.providers),
                "route": [asdict(r) for r in route],
                "version": self.coordinator.assignment_version,
                "total_layers": self.coordinator.total_layers
            }

        # Expose coordinator code generation via API
        @self.api_server.app.post("/code")
        async def generate_code(request: Dict[str, Any]):
            admin_secret = request.get("admin_secret")
            if admin_secret != os.environ.get("DICAI_ADMIN_SECRET", "admin"):
                from fastapi import HTTPException
                raise HTTPException(status_code=401, detail="Invalid admin secret")
            code = self.coordinator.auth.create_code()
            return {"code": code}

        # List codes
        @self.api_server.app.get("/codes")
        async def list_codes(admin_secret: str):
            if admin_secret != os.environ.get("DICAI_ADMIN_SECRET", "admin"):
                from fastapi import HTTPException
                raise HTTPException(status_code=401, detail="Invalid admin secret")
            return {"codes": self.coordinator.auth.list_codes()}

        # Register provider
        @self.api_server.app.post("/register")
        async def register(request: Dict[str, Any]):
            code = request.get("code") or ""
            provider_data = request.get("provider", {})
            if not self.coordinator.auth.validate_code(code):
                from fastapi import HTTPException
                raise HTTPException(status_code=403, detail="Invalid or used invite code")
            try:
                from src.dht.discovery import ProviderInfo
                provider = ProviderInfo.from_dict(provider_data)
                with self.coordinator.dht.lock:
                    self.coordinator.dht.providers[provider.provider_id] = provider
                self.coordinator.auth.mark_code_used(code, provider.provider_id)
                return {"status": "registered", "provider_id": provider.provider_id}
            except Exception as e:
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail=str(e))

    def start(self):
        print("=" * 60)
        print("DiCAI Admin Service Starting")
        print("=" * 60)

        # Start coordinator
        t1 = threading.Thread(target=self.coordinator.start, daemon=True)
        t1.start()
        time.sleep(2)

        # Start layer server
        t2 = threading.Thread(target=self.layer_server.run, daemon=True)
        t2.start()
        time.sleep(2)

        # Start API server (this blocks)
        print(f"[Admin] Web UI available at http://{self.host}:{self.api_port}")
        self.api_server.run()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DiCAI Admin Service")
    parser.add_argument("--layers-dir", default="layers", help="Directory containing layer files")
    parser.add_argument("--coordinator-port", type=int, default=8464, help="Coordinator port")
    parser.add_argument("--layer-port", type=int, default=9000, help="Layer server port")
    parser.add_argument("--api-port", type=int, default=8080, help="API/web UI port")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    args = parser.parse_args()

    service = AdminService(
        layers_dir=args.layers_dir,
        coordinator_port=args.coordinator_port,
        layer_port=args.layer_port,
        api_port=args.api_port,
        host=args.host
    )
    service.start()


if __name__ == "__main__":
    main()
