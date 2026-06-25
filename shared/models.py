from pydantic import BaseModel, Field
from typing import Optional, Literal

class ProviderRegisterRequest(BaseModel):
    device_id: str
    device_memory: float = Field(..., ge=1, description="Available memory in GB")
    memory_type: Literal["vram", "ram", "unified"] = "ram"
    compute_backend: Literal["cuda", "metal", "directml", "vulkan", "cpu"] = "cpu"
    compute_flops: Optional[float] = None
    os_type: Literal["windows", "linux", "mac"] = "linux"
    network_bandwidth: int = 1000
    gpu_name: Optional[str] = None
    cpu_cores: Optional[int] = None
    invite_token: str = Field(..., description="Admin-generated invite token")

class Provider(BaseModel):
    device_id: str
    device_memory: float
    memory_type: str = "ram"
    compute_backend: str = "cpu"
    compute_flops: Optional[float] = None
    os_type: str = "linux"
    network_bandwidth: int = 1000
    gpu_name: Optional[str] = None
    cpu_cores: Optional[int] = None
    status: str = "registered"
    last_seen: float = 0.0

class ModelSpec(BaseModel):
    model_id: str
    precision: str = "bf16"
    context_length: int = 4096
    status: str = "pending"
