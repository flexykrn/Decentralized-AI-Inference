import threading
import time
import sys
import requests
from src.admin.layer_server import LayerServer
from src.coordinator.main import Coordinator
from src.provider.main import ProviderService

p1 = None
p2 = None

def start_multi_provider():
    global p1, p2
    # Start coordinator
    coord = Coordinator(port=8464, host='0.0.0.0')
    t1 = threading.Thread(target=coord.start, daemon=True)
    t1.start()
    time.sleep(2)

    # Start layer server
    ls = LayerServer('layers', port=9000, host='0.0.0.0')
    t2 = threading.Thread(target=ls.run, daemon=True)
    t2.start()
    time.sleep(2)

    # p1: will be assigned first half
    p1 = ProviderService('p1', 'http://localhost:8464', 'http://localhost:9000', port=5001, host='0.0.0.0')
    t3 = threading.Thread(target=p1.start, daemon=True)
    t3.start()
    time.sleep(2)

    # p2: will be assigned second half
    p2 = ProviderService('p2', 'http://localhost:8464', 'http://localhost:9000', port=5002, host='0.0.0.0')
    t4 = threading.Thread(target=p2.start, daemon=True)
    t4.start()

    # Wait for both providers to load model
    for i in range(180):
        try:
            h1 = requests.get('http://localhost:5001/health', timeout=3).json()
            h2 = requests.get('http://localhost:5002/health', timeout=3).json()
            if h1.get('model_loaded') and h2.get('model_loaded'):
                print('Both providers loaded')
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print('Timeout waiting for providers to load')

    # Give coordinator rebalancing time
    time.sleep(15)

    print('p1 health:', requests.get('http://localhost:5001/health', timeout=3).json())
    print('p2 health:', requests.get('http://localhost:5002/health', timeout=3).json())
    print('ROUTE:', requests.get('http://localhost:8464/route', timeout=10).json())
    print('p1 forward:', requests.post('http://localhost:5001/forward', json={'prompt': 'Hello', 'max_tokens': 5}, timeout=30).json())
    print('p2 forward:', requests.post('http://localhost:5002/forward', json={'prompt': 'Hello', 'max_tokens': 5}, timeout=30).json())

    # Keep alive
    while True:
        time.sleep(10)

if __name__ == '__main__':
    start_multi_provider()
