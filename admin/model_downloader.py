import os
import hashlib
import json
from typing import Dict, List, Optional
from pathlib import Path

from shared.constants import MODEL_METADATA

class ModelDownloader:
    """Downloads models from HuggingFace and prepares shards for distribution."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.download_status: Dict[str, dict] = {}
    
    def get_model_path(self, model_id: str) -> Path:
        """Get local path for model."""
        return self.models_dir / model_id
    
    def download(self, model_id: str, precision: str = "bf16") -> bool:
        """Download model from HuggingFace."""
        if model_id not in MODEL_METADATA:
            print(f"Unknown model: {model_id}")
            return False
        
        model_path = self.get_model_path(model_id)
        if model_path.exists():
            print(f"Model {model_id} already downloaded")
            return True
        
        try:
            from huggingface_hub import snapshot_download
            
            print(f"Downloading {model_id}...")
            # Map our model IDs to HuggingFace repo IDs
            repo_map = {
                "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
                "llama-3-70b": "meta-llama/Meta-Llama-3-70B",
                "llama-3-405b": "meta-llama/Meta-Llama-3-405B",
                "mistral-7b": "mistralai/Mistral-7B-v0.1",
            }
            
            repo_id = repo_map.get(model_id, model_id)
            
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            
            self.download_status[model_id] = {
                "status": "completed",
                "path": str(model_path),
                "size_gb": self._get_dir_size(model_path),
            }
            
            print(f"Downloaded {model_id} to {model_path}")
            return True
            
        except Exception as e:
            print(f"Download failed: {e}")
            self.download_status[model_id] = {
                "status": "failed",
                "error": str(e),
            }
            return False
    
    def create_shards(self, model_id: str, assignments: List[dict]) -> List[dict]:
        """Create model shards for each provider based on layer assignments."""
        model_path = self.get_model_path(model_id)
        if not model_path.exists():
            print(f"Model {model_id} not downloaded")
            return []
        
        shards = []
        for assignment in assignments:
            provider_id = assignment["provider_id"]
            layer_start, layer_end = assignment["layers"]
            
            # Create shard metadata
            shard = {
                "provider_id": provider_id,
                "model_id": model_id,
                "layers": (layer_start, layer_end),
                "layer_count": assignment["layer_count"],
                "shard_id": f"{model_id}_{layer_start}_{layer_end}",
                "files": self._get_shard_files(model_path, layer_start, layer_end),
            }
            shards.append(shard)
        
        return shards
    
    def _get_shard_files(self, model_path: Path, layer_start: int, layer_end: int) -> List[str]:
        """Get list of files needed for a shard."""
        # For now, return all files (will be refined based on actual model format)
        files = []
        for f in model_path.rglob("*"):
            if f.is_file():
                files.append(str(f.relative_to(model_path)))
        return files
    
    def _get_dir_size(self, path: Path) -> float:
        """Get directory size in GB."""
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return round(total / (1024**3), 2)
    
    def get_shard_download_url(self, model_id: str, shard_id: str, base_url: str = "http://localhost:8080") -> str:
        """Get URL for provider to download shard."""
        return f"{base_url}/api/v3/models/{model_id}/shards/{shard_id}/download"

# Global instance
model_downloader = ModelDownloader()
