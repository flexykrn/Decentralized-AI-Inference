from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from typing import Dict, List, Optional
import asyncio
import time
import secrets
import os

from shared.models import Provider, ProviderRegisterRequest, ModelSpec
from shared.constants import MODEL_METADATA
from admin.calculator import calculate_model_memory
from admin.matcher import match_providers_to_model
from admin.token_store import TokenStore

app = FastAPI(title="DiCAI Admin", version="3.0")

# In-memory stores (replace with Redis/DB for production)
models_db: Dict[str, ModelSpec] = {}
providers_db: Dict[str, Provider] = {}
clusters_db: Dict[str, dict] = {}
token_store = TokenStore()

# Admin endpoints (protected by secret)
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "changeme")

@app.post("/api/v3/admin/tokens")
async def generate_token(admin_secret: str = Header(...)):
    """Generate invite token for providers."""
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    
    token = secrets.token_urlsafe(32)
    token_store.create(token)
    return {"token": token, "status": "active"}

@app.get("/api/v3/admin/tokens")
async def list_tokens(admin_secret: str = Header(...)):
    """List active invite tokens."""
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    
    return {"tokens": token_store.list_active(), "count": len(token_store.list_active())}

@app.post("/api/v3/models/{model_id}/deploy")
async def deploy_model(model_id: str, precision: str = "bf16", context_length: int = 4096):
    if model_id not in MODEL_METADATA:
        raise HTTPException(status_code=404, detail="Model not found")
    
    req = calculate_model_memory(model_id, precision, context_length)
    
    models_db[model_id] = ModelSpec(
        model_id=model_id,
        precision=precision,
        context_length=context_length,
    )
    clusters_db[model_id] = {
        "model_id": model_id,
        "precision": precision,
        "context_length": context_length,
        "providers": [],
        "status": "forming",
        "assignments": [],
        "endpoint": None,
    }
    
    return {
        "model_id": model_id,
        "precision": precision,
        "layers": req["layers"],
        "memory_per_layer_gb": req["memory_per_layer_gb"],
        "total_memory_needed_gb": req["total_memory_gb"],
        "status": "pending",
        "message": "Waiting for providers to join",
    }

@app.post("/api/v3/providers/register")
async def register_provider(req: ProviderRegisterRequest):
    # Validate invite token
    if not token_store.validate(req.invite_token):
        raise HTTPException(status_code=403, detail="Invalid or expired invite token")
    
    provider = Provider(
        device_id=req.device_id,
        device_memory=req.device_memory,
        memory_type=req.memory_type,
        compute_backend=req.compute_backend,
        compute_flops=req.compute_flops,
        os_type=req.os_type,
        network_bandwidth=req.network_bandwidth,
        gpu_name=req.gpu_name,
        cpu_cores=req.cpu_cores,
        last_seen=time.time(),
    )
    providers_db[req.device_id] = provider
    
    # Check if any pending model can now run
    matched = []
    for model_id, cluster in clusters_db.items():
        if cluster["status"] != "forming":
            continue
        
        healthy = [p for p in providers_db.values() if p.status in ("registered", "ready", "active")]
        match = match_providers_to_model(model_id, cluster["precision"], cluster["context_length"], healthy)
        
        if match["can_run"]:
            cluster["status"] = "ready"
            cluster["providers"] = [a["provider_id"] for a in match["assignments"]]
            cluster["assignments"] = match["assignments"]
            models_db[model_id].status = "ready"
            matched.append({"model_id": model_id, "providers_needed": match["providers_used"]})
    
    return {
        "device_id": req.device_id,
        "status": "registered",
        "matched_models": matched,
    }

@app.get("/api/v3/clusters/{model_id}/status")
async def cluster_status(model_id: str):
    if model_id not in clusters_db:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    cluster = clusters_db[model_id]
    healthy = [p for p in providers_db.values() if p.status in ("registered", "ready", "active")]
    match = match_providers_to_model(model_id, cluster["precision"], cluster["context_length"], healthy)
    
    return {
        "model_id": model_id,
        "cluster_status": cluster["status"],
        "can_run": match["can_run"],
        "providers_available": match["providers_available"],
        "providers_used": match["providers_used"],
        "assignments": match["assignments"] if match["can_run"] else None,
        "ready_to_start": match["can_run"] and cluster["status"] == "ready",
        "endpoint": cluster["endpoint"],
    }

@app.post("/api/v3/clusters/{model_id}/start")
async def start_cluster(model_id: str, background_tasks: BackgroundTasks):
    if model_id not in clusters_db:
        raise HTTPException(status_code=404, detail="Cluster not found")
    
    cluster = clusters_db[model_id]
    healthy = [p for p in providers_db.values() if p.status in ("registered", "ready", "active")]
    match = match_providers_to_model(model_id, cluster["precision"], cluster["context_length"], healthy)
    
    if not match["can_run"]:
        raise HTTPException(status_code=400, detail="Cannot run model with current providers")
    
    cluster["status"] = "starting"
    background_tasks.add_task(start_inference_cluster, model_id)
    
    return {
        "model_id": model_id,
        "status": "starting",
        "providers": match["providers_used"],
        "assignments": match["assignments"],
    }

async def start_inference_cluster(model_id: str):
    """Start inference cluster with assigned providers."""
    cluster = clusters_db[model_id]
    assignments = cluster.get("assignments", [])
    
    # Try distributed-llama first, fallback to llama.cpp
    try:
        # TODO: Launch distributed-llama workers for each provider
        # distributed-llama worker --model {model_id} --layers {start}-{end}
        print(f"Starting distributed-llama cluster for {model_id}")
        await asyncio.sleep(1)
    except Exception as e:
        print(f"distributed-llama failed: {e}, trying llama.cpp")
        # TODO: Launch llama.cpp servers for each provider
        # llama-cpp-python server --model {shard_path}
        await asyncio.sleep(1)
    
    cluster["status"] = "running"
    cluster["endpoint"] = f"http://localhost:9000/v1/chat/completions"
    models_db[model_id].status = "running"
    print(f"Cluster {model_id} running at {cluster['endpoint']}")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "dicai-admin-v3"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
