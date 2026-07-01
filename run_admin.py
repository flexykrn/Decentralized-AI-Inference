import threading
import time
import os
from src.coordinator.main import Coordinator
from src.admin.layer_server import LayerServer

os.environ.setdefault('DICAI_ADMIN_SECRET', 'admin')

def start_admin():
    coord = Coordinator(port=8464, host='0.0.0.0')
    t1 = threading.Thread(target=coord.start, daemon=True)
    t1.start()
    time.sleep(2)

    ls = LayerServer('layers', port=9000, host='0.0.0.0')
    t2 = threading.Thread(target=ls.run, daemon=True)
    t2.start()

    print('=== ADMIN STACK READY ===')
    while True:
        time.sleep(10)

if __name__ == '__main__':
    start_admin()
