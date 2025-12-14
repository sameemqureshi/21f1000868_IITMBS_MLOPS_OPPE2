import json, requests

URL = "http://<LB-IP-or-DNS>/predict"
with open("artifacts/random_100.json") as f:
    payload = json.load(f)[0]

r = requests.post(URL, json=payload, timeout=10)
print(r.status_code, r.json())