import threading
import time
import sys
import requests
from src.admin.layer_server import LayerServer
from src.coordinator.main import Coordinator
from src.provider.main import ProviderService

def start_stack():
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

    # Start provider
    ps = ProviderService('p1', 'http://localhost:8464', 'http://localhost:9000', port=5001, host='0.0.0.0')
    t3 = threading.Thread(target=ps.start, daemon=True)
    t3.start()
    time.sleep(25)

    print('=== STACK READY ===')
    while True:
        time.sleep(10)

if __name__ == '__main__':
    start_stack()
