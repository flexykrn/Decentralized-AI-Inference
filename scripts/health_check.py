#!/usr/bin/env python3
"""Health check script for DiCAI stack."""
import sys
import time
import requests

SERVICES = {
    "coordinator": "http://localhost:8464/health",
    "layer_server": "http://localhost:9000/manifest",
    "provider_p1": "http://localhost:5001/health",
}

def check_all(timeout=10, max_wait=120):
    start = time.time()
    statuses = {}
    while time.time() - start < max_wait:
        all_ok = True
        statuses = {}
        for name, url in SERVICES.items():
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code == 200:
                    statuses[name] = "OK"
                else:
                    statuses[name] = f"STATUS {r.status_code}"
                    all_ok = False
            except Exception as e:
                statuses[name] = f"ERROR: {e}"
                all_ok = False
        if all_ok:
            print("All services healthy:")
            for k, v in statuses.items():
                print(f"  {k}: {v}")
            return 0
        time.sleep(2)
    print("Timed out waiting for services:")
    for k, v in statuses.items():
        print(f"  {k}: {v}")
    return 1

if __name__ == "__main__":
    sys.exit(check_all())
