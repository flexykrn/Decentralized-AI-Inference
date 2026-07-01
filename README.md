# DiCAI v2 — Distributed AI Inference Network

A self-hosted distributed AI inference network. Admin seeds a model, providers join with invite codes, and users interact via an OpenAI-compatible API or web chat UI.

## What it does

- **Model splitting:** Splits a GGUF model into per-layer binary chunks.
- **Layer server:** Serves chunks to providers over HTTP.
- **Provider daemon:** Downloads assigned chunks, reassembles the model, and serves inference.
- **Coordinator:** Tracks providers, assigns layer ranges, builds inference routes.
- **API server:** OpenAI-compatible `/v1/chat/completions` endpoint.
- **Web admin UI:** Single-page dashboard for invite codes, provider monitoring, and chat.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Split a GGUF model

```bash
python3 tools/split_model.py \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --output layers \
  --n-layers 22
```

### 3. Start the admin service

```bash
export DICAI_ADMIN_SECRET="your-s...cret"
python3 -m src.web.admin_service --layers-dir layers
```

Open `http://localhost:8080` in your browser.

### 4. Add a provider

From the web UI, click **Generate Provider Code** and copy the code.

On the provider machine (same network):

```bash
python3 -m src.provider.main \
  --id p1 \
  --coordinator http://admin-ip:8464 \
  --layer-server http://admin-ip:9000 \
  --port 5001 \
  --code THE_INVITE_CODE
```

### 5. Chat

Use the web UI chat window, or call the API:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer *** \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dicai-7b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 20
  }'
```

## Ports

| Port | Service |
|------|---------|
| 8080 | Web UI + API |
| 8464 | Coordinator |
| 9000 | Layer server |
| 5001 | Default provider |

## Architecture

```
Admin Node
├── Coordinator (DHT + HTTP)
├── Layer Server
├── API Server
└── Web Dashboard

Provider Node
└── Provider Daemon
    ├── downloads assigned layer chunks
    ├── reassembles full GGUF
    └── serves /forward endpoint
```

## Current limitations

- Each provider loads the full GGUF. True per-layer computation across providers is a future milestone.
- Remote providers require a shared network path (Tailscale, public IP, or cloud VM).
- Admin UI uses a hardcoded admin secret path for demo purposes.

## Project files

| File | Purpose |
|------|---------|
| `run_demo.py` | One-command local demo stack |
| `src/web/admin_service.py` | Consolidated admin service with web UI |
| `src/provider/main.py` | Provider daemon |
| `src/coordinator/main.py` | Coordinator |
| `src/admin/layer_server.py` | Layer file server |
| `tools/split_model.py` | GGUF splitter |

## Development

To run the full stack manually:

```bash
# Terminal 1: admin service
python3 -m src.web.admin_service --layers-dir layers

# Terminal 2: provider
python3 -m src.provider.main --id p1 --coordinator http://localhost:8464 --layer-server http://localhost:9000 --port 5001 --code YOUR_CODE
```

## License

MIT
