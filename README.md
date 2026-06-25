# DiCAI - Distributed AI Inference

## What This Is

A system where:
1. **Admin** selects a model to deploy (e.g., llama-3-70b)
2. **Providers** clone the repo and run one command
3. **System** auto-detects provider hardware (GPU, RAM, OS)
4. **System** calculates if enough compute exists to run the model
5. **System** distributes model layers across providers
6. **System** generates OpenAI-compatible API endpoint
7. **When new providers join** - system redistributes load automatically

## Architecture

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Admin     │◄────────│  Provider   │◄────────│   Docker    │
│  Service    │  HTTP   │   Client    │  local  │   (exo +    │
│  (FastAPI)  │         │  (Python)   │         │  llama.cpp) │
└─────────────┘         └─────────────┘         └─────────────┘
       │                                               │
       │                                               │
       ▼                                               ▼
┌─────────────┐                              ┌─────────────┐
│  OpenAI     │                              │  Actual     │
│  Proxy      │                              │  Inference  │
│  (FastAPI)  │                              │  (exo/      │
└─────────────┘                              │  llama.cpp) │
                                             └─────────────┘
```

## Provider Flow

```bash
# Step 1: Clone repo
git clone https://github.com/flexykrn/Decentralized-AI-Inference.git

# Step 2: Run provider (ONE COMMAND)
cd Decentralized-AI-Inference
docker-compose -f docker-compose.provider.yml up

# That's it. Provider auto-detects hardware, registers with admin, waits for assignment.
```

## Admin Flow

```bash
# Step 1: Run admin
cd Decentralized-AI-Inference/admin
docker-compose up

# Step 2: Deploy model
curl -X POST http://admin-ip:8080/api/v2/models/llama-3-70b/deploy \
  -d "precision=bf16"

# Step 3: Check status
curl http://admin-ip:8080/api/v2/clusters/llama-3-70b/status

# Step 4: Start when ready
curl -X POST http://admin-ip:8080/api/v2/clusters/llama-3-70b/start

# Step 5: Use OpenAI API
curl http://admin-ip:9000/v1/chat/completions \
  -d '{"model": "llama-3-70b", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Key Features

- **Auto hardware detection**: GPU model, VRAM, RAM, OS, backend (CUDA/Metal/CPU)
- **Dynamic memory calculation**: Per-layer memory based on precision (BF16/INT8/INT4/Q4_K_M)
- **Greedy layer assignment**: Largest providers get most layers
- **Precision switching**: Same model, different precision = different provider requirements
- **Heterogeneous support**: Mix of RTX 4090, GTX 1050 Ti, Mac Mini, CPU laptops
- **Auto redistribution**: When new providers join, system recalculates and redistributes
- **Docker packaged**: Everything included, no external installs needed

## Supported Models

| Model | Params | Layers | BF16 Memory | Q4_K_M Memory |
|-------|--------|--------|-------------|---------------|
| llama-3-8b | 8B | 32 | 20GB | 5GB |
| llama-3-70b | 70B | 80 | 169GB | 38GB |
| llama-3-405b | 405B | 126 | 976GB | 220GB |

## Provider Requirements

| GPU | VRAM | Can Run (BF16) | Can Run (Q4_K_M) |
|-----|------|----------------|------------------|
| RTX 4090 | 24GB | 13 layers | 60 layers |
| RTX 3050 | 8GB | 4 layers | 20 layers |
| GTX 1050 Ti | 4GB | 2 layers | 10 layers |
| Mac Mini M2 | 16GB unified | 9 layers | 40 layers |
| CPU only | 16GB RAM | 9 layers | 40 layers |

## Development Status

- ✅ Admin service with dynamic matching
- ✅ Provider client with auto-registration
- ✅ OpenAI proxy endpoint
- ✅ Docker packaging for providers
- ✅ Hardware detection scripts
- 🔄 Exo integration (next)
- 🔄 Model download and sharding (next)
- 🔄 Real distributed inference (next)
