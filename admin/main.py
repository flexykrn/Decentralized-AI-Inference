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
from admin.model_downloader import model_downloader

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
    
    # Trigger model download and shard creation
    background_tasks.add_task(download_and_shard_model, model_id)
    
    return {
        "model_id": model_id,
        "status": "starting",
        "providers": match["providers_used"],
        "assignments": match["assignments"],
    }

async def download_and_shard_model(model_id: str):
    """Download model and create shards for providers."""
    cluster = clusters_db[model_id]
    assignments = cluster.get("assignments", [])
    
    # Download model
    success = model_downloader.download(model_id, cluster["precision"])
    if not success:
        print(f"Failed to download model {model_id}")
        return
    
    # Create shards
    shards = model_downloader.create_shards(model_id, assignments)
    cluster["shards"] = shards
    
    print(f"Created {len(shards)} shards for {model_id}")
    for shard in shards:
        print(f"  Provider {shard['provider_id']}: layers {shard['layers'][0]}-{shard['layers'][1]}")

async def start_inference_cluster(model_id: str):
    """Start inference cluster with assigned providers."""
    cluster = clusters_db[model_id]
    assignments = cluster.get("assignments", [])
    
    # Update provider assignments
    for assignment in assignments:
        provider_id = assignment["provider_id"]
        if provider_id in providers_db:
            providers_db[provider_id].status = "assigned"
            providers_db[provider_id].assigned_model = model_id
            providers_db[provider_id].layer_start = assignment["layers"][0]
            providers_db[provider_id].layer_end = assignment["layers"][1]
    
    cluster["status"] = "running"
    cluster["endpoint"] = f"http://localhost:9000/v1/chat/completions"
    models_db[model_id].status = "running"
    print(f"Cluster {model_id} running at {cluster['endpoint']}")
    
    # Start fault tolerance monitor
    asyncio.create_task(monitor_provider_health(model_id))

async def monitor_provider_health(model_id: str):
    """Monitor provider health and reassign layers if provider drops out."""
    cluster = clusters_db[model_id]
    
    while cluster["status"] == "running":
        await asyncio.sleep(10)  # Check every 10 seconds
        
        # Find offline providers
        offline_providers = []
        for assignment in cluster.get("assignments", []):
            provider_id = assignment["provider_id"]
            if provider_id in providers_db:
                provider = providers_db[provider_id]
                if provider.status == "offline" or (time.time() - provider.last_seen > 30):
                    offline_providers.append(provider_id)
                    provider.status = "offline"
        
        if not offline_providers:
            continue
        
        print(f"[Fault Tolerance] Detected {len(offline_providers)} offline providers: {offline_providers}")
        
        # Find standby providers (registered but not assigned)
        assigned_ids = {a["provider_id"] for a in cluster["assignments"]}
        standby = [p for pid, p in providers_db.items() 
                   if pid not in assigned_ids and p.status in ("registered", "active", "ready")]
        
        if not standby:
            print(f"[Fault Tolerance] No standby providers available. Cluster {model_id} degraded.")
            continue
        
        # Reassign layers from offline providers to standby
        for offline_id in offline_providers:
            # Find the assignment for this provider
            for assignment in cluster["assignments"]:
                if assignment["provider_id"] == offline_id:
                    layers = assignment["layers"]
                    layer_count = assignment["layer_count"]
                    
                    # Find best standby provider
                    best_standby = max(standby, key=lambda p: p.device_memory)
                    
                    # Reassign
                    assignment["provider_id"] = best_standby.device_id
                    print(f"[Fault Tolerance] Reassigned layers {layers[0]}-{layers[1]} from {offline_id} to {best_standby.device_id}")
                    
                    # Update provider status
                    best_standby.status = "assigned"
                    best_standby.assigned_model = model_id
                    best_standby.layer_start = layers[0]
                    best_standby.layer_end = layers[1]
                    standby.remove(best_standby)
                    
                    break

@app.get("/health")
async def health():
    return {"status": "ok", "service": "dicai-admin-v3"}

@app.post("/api/v3/providers/{device_id}/heartbeat")
async def provider_heartbeat(device_id: str):
    """Receive heartbeat from provider."""
    if device_id not in providers_db:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    provider = providers_db[device_id]
    provider.last_seen = time.time()
    provider.status = "active"
    
    return {"status": "ok", "device_id": device_id}

@app.get("/api/v3/providers")
async def list_providers():
    """List all registered providers with health status."""
    now = time.time()
    providers = []
    
    for pid, provider in providers_db.items():
        # Mark as offline if no heartbeat in 30 seconds
        if now - provider.last_seen > 30:
            provider.status = "offline"
        
        providers.append({
            "device_id": provider.device_id,
            "device_memory": provider.device_memory,
            "memory_type": provider.memory_type,
            "compute_backend": provider.compute_backend,
            "gpu_name": provider.gpu_name,
            "os_type": provider.os_type,
            "status": provider.status,
            "last_seen": provider.last_seen,
            "last_seen_ago": round(now - provider.last_seen, 1),
        })
    
    return {"providers": providers, "count": len(providers)}

@app.get("/api/v3/providers/{device_id}")
async def get_provider(device_id: str):
    """Get specific provider details."""
    if device_id not in providers_db:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    provider = providers_db[device_id]
    now = time.time()
    
    return {
        "device_id": provider.device_id,
        "device_memory": provider.device_memory,
        "memory_type": provider.memory_type,
        "compute_backend": provider.compute_backend,
        "gpu_name": provider.gpu_name,
        "os_type": provider.os_type,
        "status": provider.status,
        "last_seen": provider.last_seen,
        "last_seen_ago": round(now - provider.last_seen, 1),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
