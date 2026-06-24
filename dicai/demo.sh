#!/bin/bash

# DiCAI Distributed Inference Demo
# This script starts the admin coordinator and 15 provider nodes

set -e

ADMIN_PORT=8080
BASE_PROVIDER_PORT=50051
NUM_PROVIDERS=15
MODEL_ID="llama-3-70b"
TOTAL_LAYERS=80

echo "================================================================================"
echo "                    DiCAI Distributed Inference Demo"
echo "================================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if admin binary exists
if [ ! -f "dicai/admin/dicai-admin" ]; then
    print_warn "Admin binary not found. Building..."
    cd dicai/admin
    go build -o dicai-admin ./cmd/main.go
    cd ../..
fi

# Start admin coordinator
print_status "Starting admin coordinator on port ${ADMIN_PORT}..."
./dicai/admin/dicai-admin --addr=:${ADMIN_PORT} &
ADMIN_PID=$!
sleep 2

# Check if admin is running
if ! curl -s http://localhost:${ADMIN_PORT}/health > /dev/null; then
    print_error "Admin failed to start"
    exit 1
fi
print_status "Admin ready at http://localhost:${ADMIN_PORT}"

# Start provider nodes
print_status "Starting ${NUM_PROVIDERS} provider nodes..."
for i in $(seq 1 ${NUM_PROVIDERS}); do
    PROVIDER_ID="provider-${i}"
    PROVIDER_PORT=$((${BASE_PROVIDER_PORT} + ${i} - 1))
    
    # For demo, we'll just register providers via curl instead of running actual processes
    curl -s -X POST http://localhost:${ADMIN_PORT}/api/v1/providers/register \
        -H "Content-Type: application/json" \
        -d "{\"id\":\"${PROVIDER_ID}\",\"address\":\"localhost\",\"port\":${PROVIDER_PORT},\"memory\":16,\"backend\":\"cpu\"}" > /dev/null
    
    if [ $? -eq 0 ]; then
        print_status "Provider ${PROVIDER_ID} registered (port ${PROVIDER_PORT})"
    else
        print_error "Failed to register provider ${PROVIDER_ID}"
    fi
done

# Verify all providers registered
print_status "Verifying provider registration..."
PROVIDER_COUNT=$(curl -s http://localhost:${ADMIN_PORT}/api/v1/providers | grep -o '"id"' | wc -l)
print_status "Registered providers: ${PROVIDER_COUNT}"

# Deploy model
print_status "Deploying model ${MODEL_ID} with ${TOTAL_LAYERS} layers..."
DEPLOY_RESPONSE=$(curl -s -X POST http://localhost:${ADMIN_PORT}/api/v1/models/${MODEL_ID}/deploy \
    -H "Content-Type: application/json" \
    -d "{\"total_layers\":${TOTAL_LAYERS}}")

echo "${DEPLOY_RESPONSE}" | python3 -m json.tool 2>/dev/null || echo "${DEPLOY_RESPONSE}"

# Test inference
print_status "Testing inference..."
INFERENCE_RESPONSE=$(curl -s -X POST http://localhost:${ADMIN_PORT}/api/v1/inference \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_ID}\",\"prompt\":\"Hello, how are you?\",\"max_tokens\":50}")

echo "${INFERENCE_RESPONSE}" | python3 -m json.tool 2>/dev/null || echo "${INFERENCE_RESPONSE}"

# Test OpenAI-compatible API
print_status "Testing OpenAI-compatible API..."
CHAT_RESPONSE=$(curl -s -X POST http://localhost:${ADMIN_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-3-70b",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 50
    }')

echo "${CHAT_RESPONSE}" | python3 -m json.tool 2>/dev/null || echo "${CHAT_RESPONSE}"

# Show health summary
print_status "Health summary:"
HEALTH=$(curl -s http://localhost:${ADMIN_PORT}/api/v1/providers)
echo "${HEALTH}" | python3 -m json.tool 2>/dev/null || echo "${HEALTH}"

print_status "Demo complete!"
print_status "Admin API: http://localhost:${ADMIN_PORT}"
print_status "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "print_status 'Shutting down...'; kill ${ADMIN_PID} 2>/dev/null; exit 0" INT
wait ${ADMIN_PID}
