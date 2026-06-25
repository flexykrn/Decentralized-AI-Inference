#!/bin/bash
# DiCAI Distributed Inference - Quick Start
# 
# This script starts a distributed inference cluster on localhost.
# For production, run providers on separate machines.

set -e

MODEL="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

echo "=== DiCAI Distributed Inference - Quick Start ==="
echo ""

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    echo "Download a model first:"
    echo "  wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O $MODEL"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install --break-system-packages llama-cpp-python fastapi uvicorn 2>/dev/null || pip3 install llama-cpp-python fastapi uvicorn

echo ""
echo "Starting Provider 1 (layers 0-10) on port 8081..."
python3 provider/shard_provider.py --id p1 --model "$MODEL" --start-layer 0 --end-layer 10 --port 8081 &
P1_PID=$!

echo "Starting Provider 2 (layers 11-21) on port 8082..."
python3 provider/shard_provider.py --id p2 --model "$MODEL" --start-layer 11 --end-layer 21 --port 8082 &
P2_PID=$!

echo ""
echo "Waiting for providers to start..."
sleep 5

echo "Testing health checks..."
curl -s http://localhost:8081/health
curl -s http://localhost:8082/health

echo ""
echo "Running inference test..."
python3 coordinator/shard_coordinator.py

echo ""
echo "=== Test Complete ==="
echo "To stop providers: kill $P1_PID $P2_PID"
echo ""
echo "For production deployment:"
echo "1. Copy provider/shard_provider.py to each worker machine"
echo "2. Run with --host 0.0.0.0 --port <port>"
echo "3. Update coordinator with actual IP addresses"
