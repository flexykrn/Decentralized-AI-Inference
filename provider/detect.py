import subprocess
import platform
import json
import os
import sys
import time

def detect_hardware() -> dict:
    result = {
        "os_type": sys.platform,
        "device_id": f"{platform.node()}-{int(time.time())}",
    }
    
    # CPU cores
    result["cpu_cores"] = os.cpu_count() or 4
    
    # RAM (cross-platform)
    try:
        import psutil
        mem = psutil.virtual_memory()
        result["ram_gb"] = round(mem.total / (1024**3), 2)
    except ImportError:
        result["ram_gb"] = 16  # fallback
    
    # GPU detection
    gpu_info = detect_gpu(result)
    result.update(gpu_info)
    
    return result

def detect_gpu(result) -> dict:
    # Try nvidia-smi first
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        lines = output.strip().split("\n")
        if lines:
            parts = lines[0].split(",")
            if len(parts) >= 2:
                return {
                    "gpu_detected": True,
                    "gpu_name": parts[0].strip(),
                    "device_memory": float(parts[1].strip()) / 1024,  # MB to GB
                    "memory_type": "vram",
                    "compute_backend": "cuda",
                }
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try system_profiler on macOS
    if sys.platform == "darwin":
        try:
            output = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if "Apple" in output:
                return {
                    "gpu_detected": True,
                    "gpu_name": "Apple Silicon",
                    "device_memory": result.get("ram_gb", 16),
                    "memory_type": "unified",
                    "compute_backend": "metal",
                }
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    # Fallback to CPU
    return {
        "gpu_detected": False,
        "device_memory": result.get("ram_gb", 16),
        "memory_type": "ram",
        "compute_backend": "cpu",
    }

if __name__ == "__main__":
    print(json.dumps(detect_hardware(), indent=2))
