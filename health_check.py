import time, requests, sys

for i in range(20):
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=2)
        print(r.status_code, r.text)
        sys.exit(0)
    except Exception:
        time.sleep(0.5)

print("health check failed")
sys.exit(1)
