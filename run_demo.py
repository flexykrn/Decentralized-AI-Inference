import threading
import time
import os
import requests
from src.admin.layer_server import LayerServer
from src.coordinator.main import Coordinator
from src.provider.main import ProviderService
from src.api.server import DiCAIAPIServer

os.environ.setdefault('DICAI_ADMIN_SECRET', 'admin')

def start_demo():
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

    # Generate two invite codes: one for local provider, one for remote coworker
    codes = []
    for _ in range(2):
        r = requests.post('http://localhost:8464/code', json={'admin_secret': 'admin'}, timeout=5)
        if r.status_code != 200:
            print('Failed to generate invite code:', r.text)
            return
        codes.append(r.json()['code'])

    print('\n=== INVITE CODES ===', flush=True)
    print('LOCAL PROVIDER:', codes[0], flush=True)
    print('REMOTE COWORKER:', codes[1], flush=True)
    print('====================\n', flush=True)

    # Start local provider
    ps = ProviderService('p1', 'http://localhost:8464', 'http://localhost:9000', port=5001, host='0.0.0.0', invite_code=codes[0])
    t3 = threading.Thread(target=ps.start, daemon=True)
    t3.start()

    # Start API server
    api = DiCAIAPIServer('http://localhost:8464', port=8080, host='0.0.0.0')
    key = api.auth.create_key('demo', rate_limit=1000)
    t4 = threading.Thread(target=api.run, daemon=True)
    t4.start()

    # Wait for local provider to be ready
    for i in range(180):
        try:
            r = requests.get('http://localhost:5001/ready', timeout=3)
            if r.status_code == 200 and r.json().get('ready'):
                break
        except Exception:
            pass
        time.sleep(1)

    print('\n=== DEMO STACK READY ===', flush=True)
    print(f'API URL: http://0.0.0.0:8080/v1/chat/completions', flush=True)
    print(f'API Key: {key}', flush=True)
    print(f'Coordinator: http://0.0.0.0:8464', flush=True)
    print(f'Layer Server: http://0.0.0.0:9000', flush=True)
    print('========================\n', flush=True)

    while True:
        time.sleep(10)

if __name__ == '__main__':
    start_demo()
