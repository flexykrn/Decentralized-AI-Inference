#!/bin/bash
# Quick test script for DiCAI custom sharding

set -e

MODEL="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

echo "=== DiCAI Custom Sharding Test ==="

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    exit 1
fi

echo "Starting Provider 1 (layers 0-10)..."
python provider/shard_provider.py --id p1 --model "$MODEL" --start-layer 0 --end-layer 10 --port 8081 &
P1_PID=$!

echo "Starting Provider 2 (layers 11-21)..."
python provider/shard_provider.py --id p2 --model "$MODEL" --start-layer 11 --end-layer 21 --port 8082 &
P2_PID=$!

sleep 5

echo "Testing inference..."
python coordinator/shard_coordinator.py

echo "Cleaning up..."
kill $P1_PID $P2_PID 2>/dev/null

echo "Test complete!"
