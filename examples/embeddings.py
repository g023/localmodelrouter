#!/usr/bin/env python3
"""Embeddings example using both native and OpenAI endpoints."""
import requests

BASE = "http://localhost:11434"
MODEL = "qwen3-tiny"  # Replace with an embedding model for better results

# Native /api/embed
print("=== Native /api/embed ===")
resp = requests.post(f"{BASE}/api/embed", json={
    "model": MODEL,
    "input": "The quick brown fox jumps over the lazy dog",
})
if resp.status_code == 200:
    data = resp.json()
    embeddings = data.get("embeddings", [[]])
    if embeddings and embeddings[0]:
        print(f"  Dimensions: {len(embeddings[0])}")
        print(f"  First 5 values: {embeddings[0][:5]}")
    else:
        print("  No embeddings returned (model may not support embeddings)")
else:
    print(f"  Error: {resp.status_code} - {resp.text[:200]}")

# Native /api/embeddings (legacy)
print("\n=== Native /api/embeddings ===")
resp = requests.post(f"{BASE}/api/embeddings", json={
    "model": MODEL,
    "prompt": "Hello world",
})
if resp.status_code == 200:
    data = resp.json()
    emb = data.get("embedding", [])
    if emb:
        print(f"  Dimensions: {len(emb)}")
    else:
        print("  No embedding returned")
else:
    print(f"  Error: {resp.status_code}")

# OpenAI /v1/embeddings
print("\n=== OpenAI /v1/embeddings ===")
resp = requests.post(f"{BASE}/v1/embeddings", json={
    "model": MODEL,
    "input": ["Hello world", "Goodbye world"],
})
if resp.status_code == 200:
    data = resp.json()
    for item in data.get("data", []):
        print(f"  Index {item['index']}: {len(item['embedding'])} dimensions")
else:
    print(f"  Error: {resp.status_code}")
