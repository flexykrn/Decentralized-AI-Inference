#!/usr/bin/env python3
"""
DiCAI Shard Distribution Server

HTTP server for distributing model shards to providers.
Admin runs this, providers download shards on startup.
"""

import os
import hashlib
import json
from flask import Flask, send_file, jsonify, request, abort
from functools import wraps

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import AuthManager


class ShardServer:
    """HTTP server for distributing model shards to providers."""
    
    def __init__(self, shard_dir: str, port: int = 8080, auth_manager=None):
        self.shard_dir = shard_dir
        self.port = port
        self.auth_manager = auth_manager or AuthManager()
        self.app = Flask(__name__)
        self.setup_routes()
        
    def setup_routes(self):
        """Setup HTTP routes."""
        
        @self.app.route('/health')
        def health():
            return jsonify({"status": "ok", "service": "shard-server"})
            
        @self.app.route('/shards/<provider_id>')
        def download_shard(provider_id):
            """Download shard for a provider."""
            # Validate token
            token = request.headers.get('X-Invite-Token')
            if not token or not self.auth_manager.validate_invite_token(token):
                abort(403, "Invalid or expired invite token")
                
            shard_path = os.path.join(self.shard_dir, provider_id, "shard.safetensors")
            if not os.path.exists(shard_path):
                abort(404, f"Shard not found for {provider_id}")
                
            return send_file(shard_path, as_attachment=True)
            
        @self.app.route('/shards/<provider_id>/checksum')
        def checksum(provider_id):
            """Get SHA256 checksum for a shard."""
            shard_path = os.path.join(self.shard_dir, provider_id, "shard.safetensors")
            if not os.path.exists(shard_path):
                abort(404, f"Shard not found for {provider_id}")
                
            with open(shard_path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
                
            return jsonify({
                "provider_id": provider_id,
                "checksum": checksum,
                "size_mb": os.path.getsize(shard_path) / (1024 * 1024)
            })
            
        @self.app.route('/shards/<provider_id>/config')
        def shard_config(provider_id):
            """Get configuration for a provider shard."""
            config_path = os.path.join(self.shard_dir, provider_id, "config.json")
            if not os.path.exists(config_path):
                abort(404, f"Config not found for {provider_id}")
                
            with open(config_path) as f:
                return jsonify(json.load(f))
                
        @self.app.route('/register', methods=['POST'])
        def register_provider():
            """Register a provider with invite token."""
            data = request.json
            if not data:
                abort(400, "Missing request body")
                
            token = data.get('token')
            provider_id = data.get('provider_id')
            
            if not token or not provider_id:
                abort(400, "Missing token or provider_id")
                
            if self.auth_manager.use_invite_token(token, provider_id):
                return jsonify({
                    "status": "registered",
                    "provider_id": provider_id,
                    "message": "Provider registered successfully"
                })
            else:
                abort(403, "Invalid or expired token")
                
        @self.app.route('/admin/tokens', methods=['POST'])
        def create_token():
            """Admin: Create new invite token."""
            # Simple admin auth (in production, use proper auth)
            admin_key = request.headers.get('X-Admin-Key')
            if admin_key != os.environ.get('ADMIN_KEY', 'admin'):
                abort(403, "Invalid admin key")
                
            expires_hours = request.json.get('expires_hours', 24)
            token = self.auth_manager.generate_invite_token(expires_hours)
            
            return jsonify({
                "token": token,
                "expires_hours": expires_hours
            })
            
        @self.app.route('/admin/stats')
        def admin_stats():
            """Admin: Get system stats."""
            admin_key = request.headers.get('X-Admin-Key')
            if admin_key != os.environ.get('ADMIN_KEY', 'admin'):
                abort(403, "Invalid admin key")
                
            return jsonify({
                "tokens": self.auth_manager.list_tokens(),
                "shards_available": self._list_shards()
            })
            
    def _list_shards(self) -> list:
        """List available shards."""
        shards = []
        if os.path.exists(self.shard_dir):
            for provider_id in os.listdir(self.shard_dir):
                shard_path = os.path.join(self.shard_dir, provider_id, "shard.safetensors")
                if os.path.exists(shard_path):
                    shards.append({
                        "provider_id": provider_id,
                        "size_mb": round(os.path.getsize(shard_path) / (1024 * 1024), 2)
                    })
        return shards
        
    def run(self, debug=False):
        """Run the server."""
        print(f"[ShardServer] Starting on port {self.port}")
        print(f"[ShardServer] Shard directory: {self.shard_dir}")
        print(f"[ShardServer] Available shards: {len(self._list_shards())}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)


def test_shard_server():
    """Test shard server."""
    import tempfile
    import shutil
    
    # Create temp directory with test shard
    temp_dir = tempfile.mkdtemp()
    provider_dir = os.path.join(temp_dir, "p1")
    os.makedirs(provider_dir)
    
    # Create dummy shard
    with open(os.path.join(provider_dir, "shard.safetensors"), 'wb') as f:
        f.write(b"dummy shard data")
        
    # Create server
    auth = AuthManager(".test_server_tokens.json")
    server = ShardServer(temp_dir, port=8765, auth_manager=auth)
    
    # Test health
    with server.app.test_client() as client:
        response = client.get('/health')
        assert response.status_code == 200
        print("Health check: PASSED")
        
        # Test checksum without auth (should fail)
        response = client.get('/shards/p1/checksum')
        assert response.status_code == 200  # Public endpoint
        print("Checksum: PASSED")
        
        # Test download without auth (should fail)
        response = client.get('/shards/p1')
        assert response.status_code == 403
        print("Download without auth blocked: PASSED")
        
        # Generate token
        token = auth.generate_invite_token()
        
        # Test download with auth
        response = client.get('/shards/p1', headers={'X-Invite-Token': token})
        assert response.status_code == 200
        print("Download with auth: PASSED")
        
    # Cleanup
    shutil.rmtree(temp_dir)
    os.remove(".test_server_tokens.json")
    print("Shard server test PASSED")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DiCAI Shard Distribution Server")
    parser.add_argument("--shard-dir", required=True, help="Directory containing shards")
    parser.add_argument("--port", type=int, default=8080, help="Port")
    parser.add_argument("--test", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    if args.test:
        test_shard_server()
    else:
        server = ShardServer(args.shard_dir, args.port)
        server.run()
