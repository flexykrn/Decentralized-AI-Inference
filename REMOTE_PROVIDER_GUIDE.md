# Remote Provider Setup Guide

This guide explains how to run DiCAI with one admin machine and one remote provider machine (e.g., a coworker 60 km away).

## Architecture

- Admin machine runs: coordinator, layer server, local provider, API server.
- Remote machine runs: provider only.
- Both connect to admin's public IP.

## Prerequisites

- Python 3.11+
- Both machines have the repo cloned.
- Admin machine has a public IP and port forwarding configured (see Step 1).

## Step 1: Admin — Open Network Ports

### 1.1 Find your WSL LAN IP

```bash
ip addr show eth0 | grep "inet "
```

Example: `192.168.1.42`

### 1.2 Find your public IP

```bash
curl ifconfig.me
```

Example: `203.0.113.42`

### 1.3 Forward ports on your router

Log into your router admin panel (usually `192.168.1.1` or `192.168.0.1`).

Forward these external ports to your WSL LAN IP:

| External Port | Protocol | Internal IP | Internal Port |
|---------------|----------|-------------|---------------|
| 8464 | TCP + UDP | your-wsl-lan-ip | 8464 |
| 9000 | TCP | your-wsl-lan-ip | 9000 |
| 8080 | TCP | your-wsl-lan-ip | 8080 |

### 1.4 Open firewall on WSL

```bash
sudo ufw allow 8464/tcp
sudo ufw allow 8464/udp
sudo ufw allow 9000/tcp
sudo ufw allow 8080/tcp
```

On Windows, also allow Python through Windows Defender Firewall.

## Step 2: Admin — Start the Demo Stack

```bash
cd Decentralized-AI-Inference
python3 run_demo.py
```

This prints:
- Two invite codes.
- API URL.
- API key.

Copy the **REMOTE COWORKER** invite code and send it to your coworker.

## Step 3: Coworker — Run Provider

```bash
cd Decentralized-AI-Inference
pip install -r requirements.txt

python3 -m src.provider.main \
  --id p2 \
  --coordinator http://ADMIN_PUBLIC_IP:8464 \
  --layer-server http://ADMIN_PUBLIC_IP:9000 \
  --port 5001 \
  --code THE_INVITE_CODE
```

Replace `ADMIN_PUBLIC_IP` with the admin's public IP from Step 1.2.

## Step 4: Verify Both Providers Registered

On the admin machine:

```bash
curl http://localhost:8464/status
```

You should see two providers in the route.

## Step 5: Call the API

```bash
curl -X POST http://ADMIN_PUBLIC_IP:8080/v1/chat/completions \
  -H "Authorization: Bearer API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dicai-7b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 20
  }'
```

Replace:
- `ADMIN_PUBLIC_IP` with admin's public IP.
- `API_KEY` with the key printed by `run_demo.py`.

## Security Notes

- This exposes ports to the public internet. Use only for demos.
- The invite code is single-use. Generate a new one for each coworker.
- Set a strong `DICAI_ADMIN_SECRET` before starting the stack.

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| Coworker cannot connect | Port forwarding not working | Verify router rules and Windows firewall |
| `Invalid or used invite code` | Code already used or wrong | Generate a new code on admin |
| Coordinator shows only 1 provider | Coworker provider did not register | Check coworker provider logs |
| API returns error | No provider ready | Wait for model load, check `/ready` |

## Advanced: Run Only Admin Services (No Local Provider)

```bash
python3 run_admin.py
python3 -m src.api.server --coordinator http://localhost:8464 --port 8080
```

Then generate a code:

```bash
python3 tools/generate_provider_code.py admin
```
