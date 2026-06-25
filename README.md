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

## Current Status

**Working:**
- GGUF model parser and layer extractor
- True layer sharding (each provider gets only its layers)
- Provider HTTP API (health checks, layer serving)
- Coordinator for distributed inference routing
- Dynamic hardware-aware provider matching

**In Progress:**
- PyTorch forward pass shape alignment (RMS norm mismatch)
- End-to-end distributed inference test

**Not Started:**
- Docker packaging for providers
- Model download/sharing mechanism
- Production deployment

## Architecture

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Admin     │◄────────│  Provider   │◄────────│   PyTorch   │
│  Service    │  HTTP   │   Node      │  local  │   Shard     │
│  (FastAPI)  │         │  (Python)   │         │   Loader    │
└─────────────┘         └─────────────┘         └─────────────┘
       │                                               │
       │                                               │
       ▼                                               ▼
┌─────────────┐                              ┌─────────────┐
│  OpenAI     │                              │  Actual     │
│  Proxy      │                              │  Inference  │
│  (FastAPI)  │                              │  (PyTorch)  │
└─────────────┘                              └─────────────┘
```

## Quick Start

### For Providers

```bash
# Step 1: Clone repo
git clone https://github.com/flexykrn/Decentralized-AI-Inference.git

# Step 2: Install dependencies
pip install torch safetensors fastapi uvicorn

# Step 3: Get your shard from admin
# (Admin will give you a shard file and layer range)

# Step 4: Run provider
python3 provider/pytorch_provider.py \
  --id p1 \
  --shard-dir ./shards/provider_0 \
  --start-layer 0 \
  --end-layer 10 \
  --port 8081
```

### For Admin

```bash
# Step 1: Install dependencies
pip install torch safetensors fastapi uvicorn

# Step 2: Split model for providers
python3 tools/convert_to_pytorch.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --providers 2 \
  --output-dir pytorch_model

# Step 3: Distribute shards to providers
# (Copy pytorch_model/provider_0 to provider 1, etc.)

# Step 4: Run coordinator
python3 coordinator/shard_coordinator.py
```

## How It Works

### 1. Model Sharding

The GGUF model is split into layer groups:
- Each provider gets a subset of layers
- Embeddings go to first provider
- Output layer goes to last provider
- Each shard is ~50% of original size for 2 providers

### 2. Distributed Inference

```
User Request → Coordinator → Provider 1 (layers 0-10) → Provider 2 (layers 11-21) → Response
                    ↑                ↓                           ↓
                    └────────── Hidden States ──────────────────┘
```

### 3. Provider Node

Each provider:
- Loads only its assigned layers (~1-2GB instead of 4GB)
- Serves HTTP API for processing activations
- Receives input_ids or hidden_states from previous provider
- Returns output hidden_states to next provider

## Files

- `provider/pytorch_provider.py` - Provider node (loads shards, serves HTTP)
- `coordinator/shard_coordinator.py` - Inference router
- `tools/gguf_splitter.py` - GGUF layer extractor
- `tools/convert_to_pytorch.py` - GGUF to PyTorch converter
- `shared/distributed_model.py` - Hardware feasibility calculator

## Hardware Requirements

| Model | Precision | Providers | Per Provider | Total |
|-------|-----------|-----------|--------------|-------|
| 1.1B | Q4_K_M | 2 | ~2GB | ~4GB |
| 8B | Q4_K_M | 4 | ~2GB | ~8GB |
| 70B | Q4_K_M | 8 | ~5GB | ~40GB |
| 70B | BF16 | 16 | ~10GB | ~160GB |
| 1T | Q4_K_M | 100 | ~5GB | ~500GB |

## Known Issues

1. **Tensor shape mismatch** in PyTorch forward pass (RMS norm)
   - Embeddings output shape doesn't match layer input expectations
   - Need to fix weight transposition between GGUF and PyTorch

2. **No Docker packaging yet**
   - Providers need manual Python setup
   - Need containerized deployment

3. **No model sharing mechanism**
   - Admin needs to manually distribute shards
   - Need HTTP-based shard distribution

## Roadmap

- [x] GGUF parser and layer extractor
- [x] True layer sharding
- [x] Provider HTTP API
- [x] Coordinator routing
- [ ] Fix tensor shape mismatch
- [ ] End-to-end distributed test
- [ ] Docker packaging
- [ ] Model sharing via HTTP
- [ ] Dynamic provider joining
- [ ] 70B model test
- [ ] 1T model support

## License

MIT - Open source, free to use

## Credits

Built by DiCAI team using:
- llama.cpp (GGUF format)
- PyTorch (inference engine)
- FastAPI (HTTP API)
- safetensors (model format)
