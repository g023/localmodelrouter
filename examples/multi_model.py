#!/usr/bin/env python3
"""Multi-model usage: load different models for different tasks."""
import requests
import json

BASE = "http://localhost:11434"

models = ["g023/Qwen3-1.77B-g023-GGUF:Q8_0", "qwen3-tiny-v2"]

for model in models:
    print(f"\n=== {model} ===")
    resp = requests.post(f"{BASE}/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": "What model are you? Reply in one sentence."}],
        "stream": False,
        "options": {"num_predict": 30},
    })
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Response: {data['message']['content'][:150]}")
        print(f"  Duration: {data.get('total_duration', 0) / 1e9:.2f}s")
    else:
        print(f"  Error: {resp.status_code}")

# Show running processes
print("\n=== Running Processes ===")
resp = requests.get(f"{BASE}/api/ps")
for m in resp.json().get("models", []):
    print(f"  {m['name']} - expires: {m.get('expires_at', 'unknown')}")
