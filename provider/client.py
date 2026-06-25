import argparse
import requests
import time
import json
from provider.detect import detect_hardware

def main():
    parser = argparse.ArgumentParser(description="DiCAI Provider Client")
    parser.add_argument("--admin", default="http://localhost:8080", help="Admin URL")
    parser.add_argument("--device-id", default="auto", help="Device ID (auto=detect)")
    parser.add_argument("--model", default="", help="Model ID to host")
    parser.add_argument("--token", required=True, help="Invite token from admin")
    args = parser.parse_args()
    
    # Detect hardware
    hw = detect_hardware()
    device_id = args.device_id if args.device_id != "auto" else hw["device_id"]
    
    print(f"[Provider {device_id}] Detected hardware:")
    print(json.dumps(hw, indent=2))
    
    # Register with admin
    resp = requests.post(
        f"{args.admin}/api/v3/providers/register",
        json={
            "device_id": device_id,
            "device_memory": hw.get("device_memory", hw.get("ram_gb", 16)),
            "memory_type": hw.get("memory_type", "ram"),
            "compute_backend": hw.get("compute_backend", "cpu"),
            "os_type": hw.get("os_type", "linux"),
            "gpu_name": hw.get("gpu_name"),
            "cpu_cores": hw.get("cpu_cores"),
            "invite_token": args.token,
        },
    )
    
    if resp.status_code != 200:
        print(f"[Provider {device_id}] Registration failed: {resp.text}")
        return
    
    data = resp.json()
    print(f"[Provider {device_id}] Registered. Matched models: {data.get('matched_models', [])}")
    
    # Heartbeat loop
    while True:
        try:
            requests.post(f"{args.admin}/api/v3/providers/{device_id}/heartbeat", timeout=5)
        except Exception as e:
            print(f"[Provider {device_id}] Heartbeat failed: {e}")
        time.sleep(5)

if __name__ == "__main__":
    main()
