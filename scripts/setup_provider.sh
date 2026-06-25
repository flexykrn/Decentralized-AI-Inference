#!/bin/bash
set -e

echo "=== DiCAI Provider Setup ==="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "Docker installed. Please log out and back in."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Installing docker-compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Check for NVIDIA Docker runtime (optional)
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Checking nvidia-docker..."
    if ! docker info | grep -q nvidia; then
        echo "Installing nvidia-docker2..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt-get update && sudo apt-get install -y nvidia-docker2
        sudo systemctl restart docker
    fi
fi

echo "=== Setup Complete ==="
echo "Run: docker-compose -f docker-compose.provider.yml up"