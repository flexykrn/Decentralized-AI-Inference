import grpc
import numpy as np
import time
import asyncio
from concurrent import futures

import shared.dicai_pb2 as pb2
import shared.dicai_pb2_grpc as pb2_grpc

class TensorServicer(pb2_grpc.TensorServiceServicer):
    def __init__(self, provider_id, model_loader=None):
        self.provider_id = provider_id
        self.model_loader = model_loader
        self.request_count = 0
    
    def ForwardPass(self, request, context):
        """Process tensor through assigned layers."""
        self.request_count += 1
        
        try:
            # Deserialize tensor
            tensor = np.frombuffer(request.tensor_data, dtype=np.float32)
            tensor = tensor.reshape(request.shape)
            
            # TODO: Run actual model layers here
            # For now, just pass through (identity)
            # In real implementation:
            # output = self.model_loader.run_layers(tensor, request.layer_start, request.layer_end)
            
            output = tensor  # Placeholder
            
            return pb2.TensorResponse(
                request_id=request.request_id,
                tensor_data=output.tobytes(),
                shape=request.shape,
                dtype="float32",
                success=True,
            )
        except Exception as e:
            return pb2.TensorResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
            )
    
    def HealthCheck(self, request, context):
        import psutil
        mem = psutil.virtual_memory()
        return pb2.HealthResponse(
            status="healthy",
            device_id=self.provider_id,
            memory_used=mem.used / (1024**3),
            memory_total=mem.total / (1024**3),
        )

def start_tensor_server(provider_id, port, model_loader=None):
    """Start gRPC server for tensor passing."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_TensorServiceServicer_to_server(
        TensorServicer(provider_id, model_loader), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Tensor server started on port {port}")
    return server

async def forward_to_next_provider(tensor, next_provider, model_id, layer_start, layer_end):
    """Send tensor to next provider in pipeline."""
    channel = grpc.insecure_channel(f"{next_provider['host']}:{next_provider['port']}")
    stub = pb2_grpc.TensorServiceStub(channel)
    
    request = pb2.TensorRequest(
        model_id=model_id,
        request_id=f"req_{time.time()}",
        tensor_data=tensor.tobytes(),
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        layer_start=layer_start,
        layer_end=layer_end,
    )
    
    response = stub.ForwardPass(request)
    
    if response.success:
        output = np.frombuffer(response.tensor_data, dtype=np.float32)
        output = output.reshape(response.shape)
        return output
    else:
        raise Exception(f"Provider failed: {response.error}")
