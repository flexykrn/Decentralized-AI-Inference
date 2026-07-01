import requests
import sys

def main():
    admin_secret = sys.argv[1] if len(sys.argv) > 1 else input("Admin secret: ")
    r = requests.post('http://localhost:8464/code', json={'admin_secret': admin_secret})
    if r.status_code == 200:
        print(f"Invite code: {r.json()['code']}")
    else:
        print(f"Error: {r.status_code} {r.text}")

if __name__ == '__main__':
    main()
