# DiCAI: Decentralized AI Inference Network

Distributed AI inference across consumer hardware. 10-15 laptops collaboratively host 70B parameter models with OpenAI-compatible API.

## Architecture

```
User Request
    ↓
[Admin Coordinator - Go]
    ↓
[Provider 1] → [Provider 2] → ... → [Provider N]
    ↓
[Response]
```

Each provider holds a slice of the model. They communicate via gRPC to process inference requests collaboratively.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Admin | Go 1.22, Gin, gRPC |
| Provider P2P | Rust 1.75, libp2p |
| Inference Engine | C++20, llama.cpp backends |
| Containerization | Docker, Docker Compose |
| Communication | gRPC, Protocol Buffers |

## Quick Start

```bash
# Start admin + 15 providers
docker-compose up -d

# Test inference
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3-70b", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Scaling Roadmap](docs/SCALING_ROADMAP.md)
- [Limitations](docs/LIMITATIONS.md)
- [Deployment](docs/DEPLOYMENT.md)

## License

Apache 2.0
