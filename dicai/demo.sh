#!/bin/bash
set -e

# DiCAI Distributed Inference Demo
# This script starts the admin coordinator and 15 provider nodes

ADMIN_PORT=8080
BASE_PROVIDER_PORT=50051
NUM_PROVIDERS=15
MODEL_ID="llama-3-70b"
MODEL_LAYERS=80

echo "========================================"
echo "  DiCAI Distributed Inference Demo"
echo "========================================"
echo ""

# Check if admin is running
if ! curl -s http://localhost:$ADMIN_PORT/health > /dev/null; then
    echo "Starting admin coordinator..."
    cd /mnt/c/Users/karan/Desktop/openscans/orchestrator/dicai/admin
    ./dicai-admin &
    ADMIN_PID=$!
    echo "Admin PID: $ADMIN_PID"
    sleep 2
else
    echo "Admin already running"
fi

# Register providers
echo ""
echo "Registering $NUM_PROVIDERS providers..."
for i in $(seq 1 $NUM_PROVIDERS); do
    PORT=$((BASE_PROVIDER_PORT + i - 1))
    MEMORY=$((4 + (i % 4) * 4))  # 4, 8, 12, 16 GB
    
    curl -s -X POST http://localhost:$ADMIN_PORT/api/v1/providers/register \
        -H "Content-Type: application/json" \
        -d "{\"id\":\"p$i\",\"address\":\"localhost\",\"port\":$PORT,\"memory\":$MEMORY,\"backend\":\"cpu\"}" > /dev/null
    
    echo "  Registered provider p$i (port $PORT, ${MEMORY}GB)"
done

# Deploy model
echo ""
echo "Deploying model $MODEL_ID ($MODEL_LAYERS layers)..."
curl -s -X POST http://localhost:$ADMIN_PORT/api/v1/models/$MODEL_ID/deploy \
    -H "Content-Type: application/json" \
    -d "{\"total_layers\":$MODEL_LAYERS}"

# Show assignments
echo ""
echo "Layer assignments:"
curl -s http://localhost:$ADMIN_PORT/api/v1/models/$MODEL_ID/assignments

echo ""
echo "========================================"
echo "  Demo ready!"
echo "  Admin: http://localhost:$ADMIN_PORT"
echo "  API:   http://localhost:$ADMIN_PORT/v1/chat/completions"
echo "========================================"
