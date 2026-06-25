#!/bin/bash
# DiCAI Distributed Inference Setup
# Uses llama.cpp's built-in RPC backend for automatic model distribution

set -e

MODEL_PATH="${1:-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
NUM_PROVIDERS="${2:-2}"
BASE_PORT=50052
COORDINATOR_PORT=8080

LLAMA_SERVER="/tmp/llama.cpp/build/bin/llama-server"
RPC_SERVER="/tmp/llama.cpp/build/bin/rpc-server"

echo "=== DiCAI Distributed Inference ==="
echo "Model: $MODEL_PATH"
echo "Providers: $NUM_PROVIDERS"
echo ""

# Check binaries exist
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    echo "Build llama.cpp first: cmake -B build -DGGML_RPC=ON && cmake --build build --target llama-server rpc-server"
    exit 1
fi

if [ ! -f "$RPC_SERVER" ]; then
    echo "ERROR: rpc-server not found at $RPC_SERVER"
    exit 1
fi

# Start RPC servers (providers)
echo "Starting $NUM_PROVIDERS RPC providers..."
PIDS=()
for i in $(seq 0 $((NUM_PROVIDERS - 1))); do
    PORT=$((BASE_PORT + i))
    echo "  Starting provider $i on port $PORT..."
    $RPC_SERVER -H 0.0.0.0 -p $PORT -t 4 &
    PIDS+=($!)
    sleep 1
done

# Build RPC connection string
RPC_STRING=""
for i in $(seq 0 $((NUM_PROVIDERS - 1))); do
    PORT=$((BASE_PORT + i))
    if [ -n "$RPC_STRING" ]; then
        RPC_STRING="$RPC_STRING,localhost:$PORT"
    else
        RPC_STRING="localhost:$PORT"
    fi
done

echo ""
echo "RPC backends: $RPC_STRING"
echo ""

# Start coordinator (llama-server with RPC backends)
echo "Starting coordinator on port $COORDINATOR_PORT..."
$LLAMA_SERVER \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port $COORDINATOR_PORT \
    --ctx-size 512 \
    --rpc "$RPC_STRING" \
    --verbose

# Cleanup on exit
trap "echo 'Stopping providers...'; kill ${PIDS[@]} 2>/dev/null; exit" INT TERM
