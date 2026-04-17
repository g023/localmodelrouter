#!/usr/bin/env python3
"""Model management examples: list, show, pull, copy, delete."""
import json
import requests

BASE = "http://localhost:11434"

# List models
print("=== List Models ===")
resp = requests.get(f"{BASE}/api/tags")
for m in resp.json().get("models", []):
    print(f"  {m['name']} ({m['size'] / 1e9:.1f} GB)")

# Show model details
print("\n=== Show Model Details ===")
resp = requests.post(f"{BASE}/api/show", json={"name": "qwen3-tiny"})
if resp.status_code == 200:
    data = resp.json()
    print(f"  Model: {data.get('modelinfo', {}).get('general.architecture', 'unknown')}")
    print(f"  Parameters: {json.dumps(data.get('parameters', {}), indent=4)[:200]}")
else:
    print(f"  Error: {resp.status_code}")

# List running models
print("\n=== Running Models ===")
resp = requests.get(f"{BASE}/api/ps")
for m in resp.json().get("models", []):
    print(f"  {m['name']} - VRAM: {m.get('size_vram', 0) / 1e9:.1f} GB")

# OpenAI /v1/models
print("\n=== OpenAI Models ===")
resp = requests.get(f"{BASE}/v1/models")
for m in resp.json().get("data", []):
    print(f"  {m['id']}")
