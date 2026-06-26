from functools import wraps
import time
import grpc

class CircuitBreaker:
    """Simple circuit breaker for provider calls."""
    
    def __init__(self, failure_threshold=3, recovery_timeout=10):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
                
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
            raise e


def retry_with_backoff(max_retries=3, base_delay=1):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


class FaultTolerantClient:
    """gRPC client with retry and circuit breaker."""
    
    def __init__(self, address):
        self.address = address
        self.channel = None
        self.stub = None
        self.circuit_breaker = CircuitBreaker()
        self.connect()
        
    def connect(self):
        """Connect to provider."""
        try:
            self.channel = grpc.insecure_channel(self.address)
            grpc.channel_ready_future(self.channel).result(timeout=5)
            self.stub = dicai_pb2_grpc.ProviderServiceStub(self.channel)
            return True
        except Exception as e:
            print(f"Failed to connect to {self.address}: {e}")
            return False
            
    @retry_with_backoff(max_retries=3, base_delay=1)
    def process(self, request):
        """Process with retry and circuit breaker."""
        return self.circuit_breaker.call(self.stub.process, request)
        
    @retry_with_backoff(max_retries=3, base_delay=1)
    def health(self, request):
        """Health check with retry."""
        return self.circuit_breaker.call(self.stub.health, request)
